#include "covins_backend/map_be.hpp"

// C++
#include <dirent.h>

// UDP signaling
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <sstream>
#include <poll.h>

// COVINS
#include "covins_backend/keyframe_be.hpp"
#include "covins_backend/landmark_be.hpp"
#include "covins_backend/kf_database.hpp"
#include "covins_backend/optimization_be.hpp"

// ------------------------------------------------------------------
// SHARE monitoring server configuration
// Update these values to match your deployment.
// ------------------------------------------------------------------
#define SHARE_MONITOR_IP            "127.0.0.1"  // IP of the monitoring server
#define SHARE_MERGE_SIGNAL_PORT     9955          // merge_signal_port in main.py
#define SHARE_MERGE_APPROVAL_PORT   9956          // merge_approval_port in main.py
#define SHARE_MERGE_APPROVAL_TIMEOUT_SEC 1.0      // paper §4.4 operates under <20 ms AR constraint;
                                                  // 1 s is ample without blocking Run() excessively
// ------------------------------------------------------------------

namespace covins {

// ------------------------------------------------------------------
// Internal helpers (static = file-scope, no external linkage)
// ------------------------------------------------------------------

/**
 * Send a JSON merge event notification to the monitoring server via UDP.
 * Fails silently to avoid disrupting the SLAM process.
 */
static void SendMergeSignal(const std::string&    event_type,
                             const std::set<size_t>& agent_ids,
                             size_t kf_count = 0,
                             size_t lm_count = 0)
{
    try {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) return;

        struct sockaddr_in srv{};
        srv.sin_family      = AF_INET;
        srv.sin_port        = htons(SHARE_MERGE_SIGNAL_PORT);
        srv.sin_addr.s_addr = inet_addr(SHARE_MONITOR_IP);

        std::stringstream msg;
        msg << "{\"event\":\"" << event_type << "\","
            << "\"timestamp\":"
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count()
            << ",\"agents\":[";

        bool first = true;
        for (const auto& id : agent_ids) {
            if (!first) msg << ",";
            msg << id;
            first = false;
        }
        msg << "]";

        if (event_type == "merge_complete") {
            msg << ",\"keyframes\":" << kf_count
                << ",\"landmarks\":" << lm_count;
        }
        msg << "}";

        std::string s = msg.str();
        sendto(sock, s.c_str(), s.size(), 0,
               reinterpret_cast<struct sockaddr*>(&srv), sizeof(srv));
        close(sock);

        std::cout << "[SHARE] Merge signal sent: " << s << std::endl;

    } catch (...) {
        // Silent fail — never disrupt the SLAM process
    }
}

/**
 * Request merge approval from the monitoring server (paper §4.4.2, Eq. 6).
 *
 * Sends a merge_approval_request and waits up to SHARE_MERGE_APPROVAL_TIMEOUT_SEC
 * for a response.  On any failure (socket error, timeout, parse error) the
 * function returns true (approve) to preserve SLAM correctness — a missed
 * approval is less harmful than a missed merge.
 *
 * JSON response parsing handles both compact ("approved":true) and
 * spaced ("approved": true) formatting from Python's json.dumps().
 */
static bool RequestMergeApproval(const std::set<size_t>& agent_ids,
                                  double timeout_sec = SHARE_MERGE_APPROVAL_TIMEOUT_SEC)
{
    try {
        // Socket for sending the request
        int req_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (req_sock < 0) {
            std::cout << "[SHARE] Cannot create request socket; defaulting to approval" << std::endl;
            return true;
        }

        // Socket for receiving the response (bound to a system-assigned port)
        int resp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (resp_sock < 0) {
            close(req_sock);
            std::cout << "[SHARE] Cannot create response socket; defaulting to approval" << std::endl;
            return true;
        }

        struct sockaddr_in resp_addr{};
        resp_addr.sin_family      = AF_INET;
        resp_addr.sin_addr.s_addr = INADDR_ANY;
        resp_addr.sin_port        = 0;   // let the OS choose

        if (bind(resp_sock, reinterpret_cast<struct sockaddr*>(&resp_addr),
                 sizeof(resp_addr)) < 0) {
            close(req_sock); close(resp_sock);
            std::cout << "[SHARE] Cannot bind response socket; defaulting to approval" << std::endl;
            return true;
        }

        socklen_t len = sizeof(resp_addr);
        if (getsockname(resp_sock, reinterpret_cast<struct sockaddr*>(&resp_addr), &len) < 0) {
            close(req_sock); close(resp_sock);
            std::cout << "[SHARE] Cannot get response port; defaulting to approval" << std::endl;
            return true;
        }
        int response_port = ntohs(resp_addr.sin_port);

        // Build and send the approval request
        struct sockaddr_in srv{};
        srv.sin_family      = AF_INET;
        srv.sin_port        = htons(SHARE_MERGE_APPROVAL_PORT);
        srv.sin_addr.s_addr = inet_addr(SHARE_MONITOR_IP);

        std::stringstream req_msg;
        req_msg << "{\"event\":\"merge_approval_request\","
                << "\"timestamp\":"
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch()).count()
                << ",\"response_port\":" << response_port
                << ",\"agents\":[";
        bool first = true;
        for (const auto& id : agent_ids) {
            if (!first) req_msg << ",";
            req_msg << id;
            first = false;
        }
        req_msg << "]}";

        std::string req_str = req_msg.str();
        if (sendto(req_sock, req_str.c_str(), req_str.size(), 0,
                   reinterpret_cast<struct sockaddr*>(&srv), sizeof(srv)) < 0) {
            close(req_sock); close(resp_sock);
            std::cout << "[SHARE] Failed to send approval request; defaulting to approval" << std::endl;
            return true;
        }
        std::cout << "[SHARE] Approval request sent: " << req_str << std::endl;

        // Wait for response with timeout
        struct pollfd pfd{ resp_sock, POLLIN, 0 };
        int timeout_ms  = static_cast<int>(timeout_sec * 1000);
        int poll_result = poll(&pfd, 1, timeout_ms);

        bool approved = true;   // default: approve on any failure

        if (poll_result > 0 && (pfd.revents & POLLIN)) {
            char buf[1024];
            ssize_t n = recv(resp_sock, buf, sizeof(buf) - 1, 0);
            if (n > 0) {
                buf[n] = '\0';
                std::string resp(buf);
                std::cout << "[SHARE] Approval response received: " << resp << std::endl;

                // Parse "approved": true/false  —  handles both compact and spaced JSON
                // (Python json.dumps() produces "approved": true with a space)
                bool found = false;
                for (const auto& pat : {"\"approved\":true",  "\"approved\": true"}) {
                    if (resp.find(pat) != std::string::npos) {
                        approved = true; found = true; break;
                    }
                }
                if (!found) {
                    for (const auto& pat : {"\"approved\":false", "\"approved\": false"}) {
                        if (resp.find(pat) != std::string::npos) {
                            approved = false; found = true; break;
                        }
                    }
                }
                if (!found) {
                    std::cout << "[SHARE] Unrecognised response format; defaulting to approval" << std::endl;
                    approved = true;
                }

            } else {
                std::cout << "[SHARE] Empty response; defaulting to approval" << std::endl;
            }

        } else if (poll_result == 0) {
            std::cout << "[SHARE] Approval timeout (" << timeout_sec
                      << "s); defaulting to approval" << std::endl;
        } else {
            std::cout << "[SHARE] Poll error; defaulting to approval" << std::endl;
        }

        close(req_sock);
        close(resp_sock);
        return approved;

    } catch (const std::exception& e) {
        std::cout << "[SHARE] Exception in RequestMergeApproval: " << e.what()
                  << "; defaulting to approval" << std::endl;
        return true;
    } catch (...) {
        std::cout << "[SHARE] Unknown error in RequestMergeApproval; defaulting to approval" << std::endl;
        return true;
    }
}

// ------------------------------------------------------------------
// MapInstance / MapManager
// ------------------------------------------------------------------

MapInstance::MapInstance(int id) {
    map.reset(new Map(id));
}

MapInstance::MapInstance(MapInstancePtr map_target, MapInstancePtr map_tofuse,
                         TransformType T_wmatch_wtofuse)
    : usage_cnt(-1)
{
    map.reset(new Map(map_target->map, map_tofuse->map, T_wmatch_wtofuse));
}

MapInstance::MapInstance(MapPtr external_map) {
    map = external_map;
}

MapManager::MapManager(VocabularyPtr voc)
    : voc_(voc)
{
    if(covins_params::placerec::type == "COVINS" ||
       covins_params::placerec::type == "COVINS_G") {
        database_.reset(new KeyframeDatabase(voc_));
    } else {
        std::cout << COUTFATAL << "Place Recognition Type \""
                  << covins_params::placerec::type << "\" not valid" << std::endl;
        exit(-1);
    }
    if(!voc_) {
        std::cout << COUTFATAL << "invalid vocabulary ptr" << std::endl;
        exit(-1);
    }
}

auto MapManager::AddToDatabase(KeyframePtr kf)->void {
    std::unique_lock<std::mutex> lock(mtx_access_);
    database_->AddKeyframe(kf);
}

auto MapManager::CheckMergeBuffer()->bool {
    std::unique_lock<std::mutex> lock(mtx_access_);
    return !(buffer_merge_.empty());
}

auto MapManager::CheckoutMap(int map_id, int &check_num)->MapPtr {
    std::unique_lock<std::mutex> lock(mtx_access_);
    MapContainer::iterator mit = maps_.find(map_id);
    if(mit == maps_.end()){
        std::cout << COUTERROR << "no existing map with Map-ID " << map_id << std::endl;
        return nullptr;
    }
    MapInstancePtr map = mit->second;
    if(map->usage_cnt == -1 || map->block_checkout) return nullptr;
    int check = rand();
    map->usage_cnt++;
    map->check_nums.insert(check);
    check_num = check;
    return map->map;
}

auto MapManager::CheckoutMapExclusive(int map_id, int &check_num)->MapPtr {
    std::unique_lock<std::mutex> lock(mtx_access_);
    MapContainer::iterator mit = maps_.find(map_id);
    if(mit == maps_.end()){
        std::cout << COUTFATAL << "no existing map with Map-ID " << map_id << std::endl;
        return nullptr;
    }
    MapInstancePtr map = mit->second;
    if(map->usage_cnt != 0) return nullptr;
    int check = rand();
    map->usage_cnt = -1;
    map->check_nums.insert(check);
    map->block_checkout = false;
    check_num = check;
    return map->map;
}

auto MapManager::CheckoutMapExclusiveOrWait(int map_id, int &check_num)->MapPtr {
    {
        MapContainer::iterator mit = maps_.find(map_id);
        if(mit == maps_.end()){
            std::cout << COUTFATAL << "no existing map with Map-ID " << map_id << std::endl;
            return nullptr;
        }
    }
    MapPtr map = this->CheckoutMapExclusive(map_id, check_num);
    if(!map) {
        while(!this->SetCheckoutBlock(map_id, true)) usleep(100);
    }
    while(!map) {
        map = this->CheckoutMapExclusive(map_id, check_num);
        usleep(100);
    }
    return map;
}

auto MapManager::CheckoutMapOrWait(int map_id, int &check_num)->MapPtr {
    {
        MapContainer::iterator mit = maps_.find(map_id);
        if(mit == maps_.end()){
            std::cout << COUTFATAL << "no existing map with Map-ID " << map_id << std::endl;
            return nullptr;
        }
    }
    MapPtr map = this->CheckoutMap(map_id, check_num);
    while(!map) {
        map = this->CheckoutMap(map_id, check_num);
        usleep(100);
    }
    return map;
}

auto MapManager::EraseFromDatabase(KeyframePtr kf)->void {
    std::unique_lock<std::mutex> lock(mtx_access_);
    database_->EraseKeyframe(kf);
}

auto MapManager::GetDatabase()->DatabasePtr {
    std::unique_lock<std::mutex> lock(mtx_access_);
    return database_;
}

auto MapManager::InitializeMap(int map_id)->void {
    std::unique_lock<std::mutex> lock(mtx_access_);
    if(maps_.find(map_id) != maps_.end()){
        std::cout << COUTFATAL << "Existing map with Map-ID " << map_id << std::endl;
        exit(-1);
    }
    maps_[map_id] = MapInstancePtr(new MapInstance(map_id));
}

auto MapManager::PerformMerge()->void {

    std::cout << "+++ Perform Merge +++" << std::endl;

    KeyframePtr kf_query, kf_match;
    TransformType T_smatch_squery;
    TypeDefs::Matrix6Type cov_mat;

    std::cout << "--> Fetch merge data" << std::endl;
    {
        std::unique_lock<std::mutex> lock(mtx_access_);
        MergeInformation merge = buffer_merge_.front();
        buffer_merge_.pop_front();
        kf_query        = merge.kf_query;
        kf_match        = merge.kf_match;
        T_smatch_squery = merge.T_smatch_squery;
        cov_mat         = merge.cov_mat;
    }
    // NOTE: the merge request has now been popped from the buffer.
    // If the merge is denied below, it is intentionally discarded —
    // this is the paper's "selective map merging" behaviour (§4.4.2).

    std::cout << "--> Process merge data" << std::endl;

    // *** SHARE §4.4.2: request approval from monitoring server ***
    std::set<size_t> merging_agents = {kf_query->id_.second, kf_match->id_.second};
    bool merge_approved = RequestMergeApproval(merging_agents);

    if (!merge_approved) {
        std::cout << "\033[1;31m[SHARE] Merge SKIPPED (agents "
                  << kf_query->id_.second << "," << kf_match->id_.second
                  << ") — P_skip criterion triggered\033[0m" << std::endl;
        return;   // intentional discard per paper §4.4.2
    }

    std::cout << "\033[1;32m[SHARE] Merge APPROVED — proceeding\033[0m" << std::endl;
    SendMergeSignal("merge_start", merging_agents);

    int check_query, check_match;
    this->CheckoutMapExclusiveOrWait(kf_query->id_.second, check_query);
    MapInstancePtr map_query = maps_[kf_query->id_.second];
    this->CheckoutMapExclusiveOrWait(kf_match->id_.second, check_match);
    MapInstancePtr map_match = maps_[kf_match->id_.second];

    std::cout << "----> Merge maps" << std::endl;

    TransformType T_w_squery       = kf_query->GetPoseTws();
    TransformType T_w_smatch       = kf_match->GetPoseTws();
    TransformType T_wmatch_wquery  = T_w_smatch * T_smatch_squery * T_w_squery.inverse();

    MapInstancePtr map_merged(new MapInstance(map_match, map_query, T_wmatch_wquery));

    LoopConstraint lc(kf_match, kf_query, T_smatch_squery, cov_mat);
    map_merged->map->AddLoopConstraint(lc);

    std::cout << "Merged Map: Agents|KFs|LMs: "
              << map_merged->map->associated_clients_.size() << "|"
              << map_merged->map->GetKeyframes().size()      << "|"
              << map_merged->map->GetLandmarks().size()      << std::endl;

    for(auto sit = map_merged->map->associated_clients_.begin();
        sit != map_merged->map->associated_clients_.end(); ++sit) {
        maps_[*sit] = map_merged;
    }
    map_merged->usage_cnt = 0;

    SendMergeSignal("merge_complete",
                    map_merged->map->associated_clients_,
                    map_merged->map->GetKeyframes().size(),
                    map_merged->map->GetLandmarks().size());

    std::cout << "\033[1;32m+++ MAPS MERGED +++\033[0m" << std::endl;
}

auto MapManager::RegisterMap(MapPtr external_map, bool enter_kfs_in_database)->bool {
    if(enter_kfs_in_database) {
        auto keyframes = external_map->GetKeyframesVec();
        for(const auto& kf : keyframes) this->AddToDatabase(kf);
    } else {
        std::cout << COUTNOTICE << "!!! KFs not entered in database !!!" << std::endl;
    }
    MapInstancePtr map_inst(new MapInstance(external_map));
    for(auto sit = map_inst->map->associated_clients_.begin();
        sit != map_inst->map->associated_clients_.end(); ++sit) {
        std::cout << "----> Add maps_[" << *sit << "]" << std::endl;
        maps_[*sit] = map_inst;
    }
    return true;
}

auto MapManager::RegisterMerge(MergeInformation merge_data)->void {
    std::unique_lock<std::mutex> lock(mtx_access_);
    buffer_merge_.push_back(merge_data);
}

auto MapManager::ReturnMap(int map_id, int &check_num)->void {
    std::unique_lock<std::mutex> lock(mtx_access_);
    MapContainer::iterator mit = maps_.find(map_id);
    if(mit == maps_.end()){
        std::cout << COUTFATAL << "no existing map with Map-ID " << map_id << std::endl;
        exit(-1);
    }
    MapInstancePtr map = mit->second;
    if(!map->check_nums.count(check_num)){
        std::cout << COUTFATAL << "check_num error" << std::endl;
        exit(-1);
    }
    if(map->usage_cnt == -1) map->usage_cnt = 0;
    else                     map->usage_cnt--;
    map->check_nums.erase(check_num);
}

auto MapManager::Run()->void {
    while(1) {
        if(this->CheckMergeBuffer()) this->PerformMerge();
        usleep(5000);
    }
}

auto MapManager::SetCheckoutBlock(int map_id, bool val)->bool {
    std::unique_lock<std::mutex> lock(mtx_access_);
    MapContainer::iterator mit = maps_.find(map_id);
    if(mit == maps_.end()){
        std::cout << COUTFATAL << "no existing map with Map-ID " << map_id << std::endl;
        exit(-1);
    }
    MapInstancePtr map = mit->second;
    if(val && map->block_checkout) return false;
    map->block_checkout = val;
    return true;
}

// ------------------------------------------------------------------
// Map
// ------------------------------------------------------------------

Map::Map(size_t id)
    : MapBase(id)
{
    associated_clients_.insert(id);
}

Map::Map(MapPtr map_target, MapPtr map_tofuse, TransformType T_wtarget_wtofuse)
    : MapBase(map_target->id_map_)
{
    std::set<size_t>    ac_target  = map_target->associated_clients_;
    KeyframeMap         kf_target  = map_target->GetKeyframes();
    LandmarkMap         lm_target  = map_target->GetLandmarks();
    size_t              maxkf_tgt  = map_target->GetMaxKfId();
    size_t              maxlm_tgt  = map_target->GetMaxLmId();
    LoopVector          lp_target  = map_target->GetLoopConstraints();

    std::set<size_t>    ac_tofuse  = map_tofuse->associated_clients_;
    KeyframeMap         kf_tofuse  = map_tofuse->GetKeyframes();
    LandmarkMap         lm_tofuse  = map_tofuse->GetLandmarks();
    size_t              maxkf_fuse = map_tofuse->GetMaxKfId();
    size_t              maxlm_fuse = map_tofuse->GetMaxLmId();
    LoopVector          lp_tofuse  = map_tofuse->GetLoopConstraints();

    associated_clients_.insert(ac_target.begin(),  ac_target.end());
    associated_clients_.insert(ac_tofuse.begin(),  ac_tofuse.end());
    keyframes_.insert(kf_target.begin(),  kf_target.end());
    keyframes_.insert(kf_tofuse.begin(),  kf_tofuse.end());
    landmarks_.insert(lm_target.begin(),  lm_target.end());
    landmarks_.insert(lm_tofuse.begin(),  lm_tofuse.end());
    max_id_kf_ = std::max(maxkf_tgt, maxkf_fuse);
    max_id_lm_ = std::max(maxlm_tgt, maxlm_fuse);
    loop_constraints_.insert(loop_constraints_.end(), lp_target.begin(), lp_target.end());
    loop_constraints_.insert(loop_constraints_.end(), lp_tofuse.begin(), lp_tofuse.end());

    for(auto& kv : kf_tofuse) {
        KeyframePtr kf     = kv.second;
        TransformType Tws  = T_wtarget_wtofuse * kf->GetPoseTws();
        kf->SetPoseTws(Tws);
        kf->velocity_ = T_wtarget_wtofuse.block<3,3>(0,0) * kf->velocity_;
    }
    Matrix3Type R = T_wtarget_wtofuse.block<3,3>(0,0);
    Vector3Type t = T_wtarget_wtofuse.block<3,1>(0,3);
    for(auto& kv : lm_tofuse) {
        LandmarkPtr lm = kv.second;
        lm->SetWorldPos(R * lm->GetWorldPos() + t);
    }
}

auto Map::AddKeyframe(KeyframePtr kf)->void { this->AddKeyframe(kf, false); }

auto Map::AddKeyframe(covins::MapBase::KeyframePtr kf, bool suppress_output)->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    keyframes_[kf->id_] = kf;
    max_id_kf_ = std::max(max_id_kf_, kf->id_.first);
    if(!suppress_output && !(keyframes_.size() % 50)) {
        std::cout << "Map " << this->id_map_ << " : "
                  << keyframes_.size() << " KFs | "
                  << landmarks_.size() << " LMs" << std::endl;
        this->WriteKFsToFile();
        this->WriteKFsToFileAllAg();
    }
}

auto Map::AddLandmark(covins::MapBase::LandmarkPtr lm)->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    landmarks_[lm->id_] = lm;
    max_id_lm_ = std::max(max_id_lm_, lm->id_.first);
}

auto Map::AddLoopConstraint(LoopConstraint lc)->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    loop_constraints_.push_back(lc);
    lc.kf1->is_loop_kf_ = true;
    lc.kf2->is_loop_kf_ = true;
}

auto Map::ApplyLoopCorrection(KeyframePtr kf_query, KeyframePtr kf_match,
                               TransformType T_smatch_squery)->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    TransformType T_w_squery_bef  = kf_match->GetPoseTws();
    TransformType T_w_smatch      = kf_match->GetPoseTws();
    TransformType T_w_squery_corr = T_w_smatch * T_smatch_squery;
    kf_query->SetPoseTws(T_w_squery_corr);

    KeyframePtr succ = kf_query->GetSuccessor();
    while(succ) {
        TransformType Tws_bef   = succ->GetPoseTws();
        TransformType T_qs_succ = T_w_squery_bef * Tws_bef.inverse();
        succ->SetPoseTws(T_w_squery_corr * T_qs_succ);
        succ = succ->GetSuccessor();
    }
    loop_constraints_.push_back(LoopConstraint(kf_match, kf_query, T_smatch_squery));
}

auto Map::ConvertToMsgFileExport(MsgMap &msg)->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    msg.id_map = id_map_;
    msg.keyframes1.reserve(loop_constraints_.size());
    msg.keyframes2.reserve(loop_constraints_.size());
    msg.transforms12.reserve(loop_constraints_.size());
    msg.cov.reserve(loop_constraints_.size());
    for(const auto& i : loop_constraints_) {
        msg.keyframes1.push_back(i.kf1->id_);
        msg.keyframes2.push_back(i.kf2->id_);
        msg.transforms12.push_back(i.T_s1_s2);
        msg.cov.push_back(i.cov_mat);
    }
}

auto Map::Clean()->void {
    std::unique_lock<std::mutex> lock(mtx_map_);
    std::cout << "+++ Clean Map +++" << std::endl;
    this->RemoveLandmarkOutliers();
    std::cout << "+++ DONE +++" << std::endl;
}

auto Map::EraseKeyframe(KeyframePtr kf, bool mtx_lock)->bool {
    if(mtx_lock) mtx_map_.lock();
    bool success = false;
    auto mit = keyframes_.find(kf->id_);
    if(mit != keyframes_.end()) {
        if(kf->SetInvalid()) {
            keyframes_.erase(mit);
            keyframes_erased_[kf->id_] = kf;
            success = true;
        } else {
            std::cout << COUTWARN << "Map " << id_map_
                      << ": could not remove " << kf << std::endl;
        }
    } else {
        std::cout << COUTWARN << "Map " << id_map_
                  << ": KF not found for erase" << std::endl;
    }
    if(mtx_lock) mtx_map_.unlock();
    return success;
}

auto Map::EraseKeyframeWithDatabase(KeyframePtr kf, bool mtx_lock,
                                     MapManager::DatabasePtr database)->bool {
    bool success = this->EraseKeyframe(kf, mtx_lock);
    if(!database) std::cout << COUTNOTICE << "need to erase KF from database!" << std::endl;
    else          database->EraseKeyframe(kf);
    return success;
}

auto Map::UpdateCovisibilityConnections(idpair kf_id)->void {
    std::unique_lock<std::mutex> lock(mtx_update_connections_);
    if(kf_id != defpair) {
        KeyframePtr kf = this->GetKeyframe(kf_id);
        if(kf) kf->UpdateCovisibilityConnections();
    } else {
        std::unique_lock<std::mutex> lock2(mtx_map_);
        for(const auto& mit : keyframes_)
            mit.second->UpdateCovisibilityConnections();
    }
}

auto Map::WriteKFsToFile(std::string suffix)->void {
    for(auto sit = associated_clients_.begin();
        sit != associated_clients_.end(); ++sit) {
        int cid = *sit;
        if(covins_params::sys::trajectory_format == "EUROC") {
            std::stringstream ss;
            ss << covins_params::sys::output_dir << "KF_" << cid << suffix << "_feuroc.csv";
            this->WriteStateToCsv(ss.str(), cid);
        } else if(covins_params::sys::trajectory_format == "TUM") {
            std::stringstream ss;
            ss << covins_params::sys::output_dir << "KF_" << cid << suffix << "_ftum.csv";
            this->WriteStateToCsvTUM(ss.str(), cid);
        } else {
            std::cout << COUTFATAL << "trajectory_format '"
                      << covins_params::sys::trajectory_format
                      << "' not in { EUROC | TUM }" << std::endl;
            exit(-1);
        }
    }
}

auto Map::WriteKFsToFileAllAg(std::string prefix)->void {
    for(auto sit = associated_clients_.begin();
        sit != associated_clients_.end(); ++sit) {
        int cid = *sit;
        std::stringstream ss;
        ss << covins_params::sys::output_dir
           << "stamped_traj_estimate" << prefix << ".txt";
        if(covins_params::sys::trajectory_format == "EUROC") {
            if(sit == associated_clients_.begin())
                this->WriteStateToCsv(ss.str(), cid);
            else
                this->WriteStateToCsv(ss.str(), cid, false);
        } else if(covins_params::sys::trajectory_format == "TUM") {
            if(sit == associated_clients_.begin())
                this->WriteStateToCsvTUM(ss.str(), cid);
            else
                this->WriteStateToCsvTUM(ss.str(), cid, false);
        } else {
            std::cout << COUTFATAL << "trajectory_format '"
                      << covins_params::sys::trajectory_format
                      << "' not in { EUROC | TUM }" << std::endl;
            exit(-1);
        }
    }
}

auto Map::WriteStateToCsv(const std::string& filename,
                           const size_t client_id, const bool trnc)->void {
    KeyframeVector fkfs;
    fkfs.reserve(keyframes_.size());
    for(auto& kv : keyframes_)
        if(kv.first.second == client_id) fkfs.push_back(kv.second);
    if(fkfs.empty()) return;
    std::sort(fkfs.begin(), fkfs.end(), Keyframe::CompStamp);

    std::ofstream fs;
    fs.open(filename, trnc ? std::ios::out | std::ios::trunc
                           : std::ios::out | std::ios::app);
    if(!fs.is_open()) {
        std::cout << COUTERROR << ": Unable to open " << filename << std::endl;
        return;
    }
    for(const auto& kf : fkfs) {
        Eigen::Vector3d ba, bg, vel = kf->GetStateVelocity();
        kf->GetStateBias(ba, bg);
        Eigen::Matrix4d Tws = kf->GetPoseTws();
        Eigen::Quaterniond q(Tws.block<3,3>(0,0));
        fs << std::setprecision(25) << kf->timestamp_ * 1e9f << ","
           << Tws(0,3) << "," << Tws(1,3) << "," << Tws(2,3) << ","
           << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ","
           << vel[0] << "," << vel[1] << "," << vel[2] << ","
           << bg[0]  << "," << bg[1]  << "," << bg[2]  << ","
           << ba[0]  << "," << ba[1]  << "," << ba[2]  << std::endl;
    }
    fs.close();
}

auto Map::WriteStateToCsvTUM(const std::string& filename,
                              const size_t client_id, const bool trnc)->void {
    KeyframeVector fkfs;
    fkfs.reserve(keyframes_.size());
    for(auto& kv : keyframes_)
        if(kv.first.second == client_id) fkfs.push_back(kv.second);
    if(fkfs.empty()) return;
    std::sort(fkfs.begin(), fkfs.end(), Keyframe::CompStamp);

    std::ofstream fs;
    fs.open(filename, trnc ? std::ios::out | std::ios::trunc
                           : std::ios::out | std::ios::app);
    if(!fs.is_open()) {
        std::cout << COUTERROR << ": Unable to open " << filename << std::endl;
        return;
    }
    for(const auto& kf : fkfs) {
        Eigen::Matrix4d Tws = kf->GetPoseTws();
        Eigen::Quaterniond q(Tws.block<3,3>(0,0));
        fs << std::setprecision(25) << kf->timestamp_ << " "
           << Tws(0,3) << " " << Tws(1,3) << " " << Tws(2,3) << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
           << std::endl;
    }
    fs.close();
}

} // namespace covins
