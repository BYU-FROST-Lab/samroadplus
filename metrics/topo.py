"""
TOPO metric — self-contained Python port.
Replaces hopcroftkarp with networkx.bipartite and rtree with scipy.spatial.cKDTree.
Ported from Sat2Graph/metrics/topo/topo.py
"""
import numpy as np
import math
import pickle
from scipy.spatial import cKDTree
import networkx as nx
from networkx.algorithms import bipartite

from .graph import RoadGraph, distance


# ── Helpers ──────────────────────────────────────────────────────────────────

def latlonNorm(p1, lat=40):
    p11 = p1[1] * math.cos(math.radians(lat))
    l = np.sqrt(p11 * p11 + p1[0] * p1[0])
    if l == 0:
        return 0, 0
    return p1[0] / l, p11 / l


def pointToLineDistance(p1, p2, p3):
    dist = np.sqrt(p2[0] * p2[0] + p2[1] * p2[1])
    if dist == 0:
        a = p3[0] - p1[0]
        b = p3[1] - p1[1]
        return np.sqrt(a * a + b * b)
    proj_length = (p2[0] * p3[0] + p2[1] * p3[1]) / dist
    if proj_length > dist:
        a = p3[0] - p2[0]
        b = p3[1] - p2[1]
        return np.sqrt(a * a + b * b)
    if proj_length < 0:
        a = p3[0] - p1[0]
        b = p3[1] - p1[1]
        return np.sqrt(a * a + b * b)
    alpha = proj_length / dist
    p4 = [alpha * p2[0], alpha * p2[1]]
    a = p3[0] - p4[0]
    b = p3[1] - p4[1]
    return np.sqrt(a * a + b * b)


def pointToLineDistanceLatLon(p1, p2, p3):
    pp2 = [p2[0] - p1[0], (p2[1] - p1[1]) * math.cos(math.radians(p1[0]))]
    pp3 = [p3[0] - p1[0], (p3[1] - p1[1]) * math.cos(math.radians(p1[0]))]
    return pointToLineDistance((0, 0), pp2, pp3)


def _hopcroft_karp_matching(bigraph_dict):
    """Use networkx bipartite matching as drop-in for HopcroftKarp."""
    G = nx.Graph()
    for marble, hole_set in bigraph_dict.items():
        marble_key = ("m", marble)
        G.add_node(marble_key, bipartite=0)
        for hole_id in hole_set:
            hole_key = ("h", hole_id)
            G.add_node(hole_key, bipartite=1)
            G.add_edge(marble_key, hole_key)
    if G.number_of_nodes() == 0:
        return {}
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    matching = bipartite.hopcroft_karp_matching(G, top_nodes)
    return matching


# ── Coordinate Conversion ───────────────────────────────────────────────────

LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0


def xy2latlon(x, y):
    lat = LAT_TOP_LEFT - x * 1.0 / 111111.0
    lon = LON_TOP_LEFT + (y * 1.0 / 111111.0) / math.cos(math.radians(LAT_TOP_LEFT))
    return lat, lon


def create_graph_from_pickle(m):
    """Convert a pickle adjacency dict {(x,y): [(x2,y2), ...]} to RoadGraph."""
    graph = RoadGraph()
    nid = 0
    idmap = {}
    min_lat = LAT_TOP_LEFT
    max_lon = LON_TOP_LEFT

    for k, v in m.items():
        n1 = k
        lat1, lon1 = xy2latlon(n1[0], n1[1])
        if lat1 < min_lat:
            min_lat = lat1
        if lon1 > max_lon:
            max_lon = lon1

        for n2 in v:
            lat2, lon2 = xy2latlon(n2[0], n2[1])

            if n1 in idmap:
                id1 = idmap[n1]
            else:
                id1 = nid
                idmap[n1] = nid
                nid += 1

            if n2 in idmap:
                id2 = idmap[n2]
            else:
                id2 = nid
                idmap[n2] = nid
                nid += 1

            graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)

    graph.ReverseDirectionLink()
    for node in graph.nodes:
        graph.nodeScore[node] = 100
    for edge in graph.edges:
        graph.edgeScore[edge] = 100

    return graph, min_lat, max_lon


# ── TOPO Starting Points ────────────────────────────────────────────────────

def TOPOGenerateStartingPoints(OSMMap, region=None, mergin=0.07):
    result = []
    visitedNodes = []

    for nodeid in OSMMap.nodes.keys():
        if nodeid in visitedNodes:
            continue
        cur_node = nodeid
        next_nodes = {}
        for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
            next_nodes[nn] = 1

        if len(next_nodes) == 2:
            continue

        for nextnode in next_nodes.keys():
            if nextnode in visitedNodes:
                continue
            node_list = [nodeid]
            cur_node = nextnode
            while True:
                node_list.append(cur_node)
                neighbor = {}
                for nn in OSMMap.nodeLink[cur_node] + OSMMap.nodeLinkReverse[cur_node]:
                    neighbor[nn] = 1
                if len(neighbor) != 2:
                    break
                _keys = list(neighbor.keys())
                if node_list[-2] == _keys[0]:
                    cur_node = _keys[1]
                else:
                    cur_node = _keys[0]

            for i in range(1, len(node_list) - 1):
                visitedNodes.append(node_list[i])

            dists = []
            dist = 0
            for i in range(len(node_list) - 1):
                dists.append(dist)
                dist += distance(OSMMap.nodes[node_list[i]],
                                 OSMMap.nodes[node_list[i + 1]])
            dists.append(dist)

            density = 0.00050
            if dist < density / 2:
                continue
            n = max(int(dist / density), 1)
            alphas = [float(x + 1) / float(n + 1) for x in range(n)]

            for alpha in alphas:
                for j in range(len(node_list) - 1):
                    if alpha * dist >= dists[j] and alpha * dist <= dists[j + 1]:
                        a = (alpha * dist - dists[j]) / (dists[j + 1] - dists[j])
                        lat = ((1 - a) * OSMMap.nodes[node_list[j]][0] +
                               a * OSMMap.nodes[node_list[j + 1]][0])
                        lon = ((1 - a) * OSMMap.nodes[node_list[j]][1] +
                               a * OSMMap.nodes[node_list[j + 1]][1])

                        if region is not None:
                            lat_mergin = mergin * (region[2] - region[0])
                            lon_mergin = mergin * (region[3] - region[1])
                            if (lat - region[0] > lat_mergin and
                                    region[2] - lat > lat_mergin and
                                    lon - region[1] > lon_mergin and
                                    region[3] - lon > lon_mergin):
                                result.append((lat, lon, node_list[j],
                                               node_list[j + 1],
                                               alpha * dist - dists[j],
                                               dists[j + 1] - alpha * dist))
    return result


# ── TOPO Pair Generation (using cKDTree) ─────────────────────────────────────

def TOPOGeneratePairs(GPSMap, OSMMap, OSMList, threshold=0.00010):
    result = {}

    # Build KDTree from GPS map edges
    edge_data = []
    edge_ids = []
    for edgeid in GPSMap.edges.keys():
        n1 = GPSMap.edges[edgeid][0]
        n2 = GPSMap.edges[edgeid][1]
        lat1, lon1 = GPSMap.nodes[n1]
        lat2, lon2 = GPSMap.nodes[n2]
        # Store midpoint for spatial indexing, plus bounding box
        edge_data.append([min(lat1, lat2), min(lon1, lon2),
                          max(lat1, lat2), max(lon1, lon2)])
        edge_ids.append(edgeid)

    if not edge_ids:
        return result

    # Build a KDTree of all edge midpoints for quick spatial lookup
    midpoints = np.array([[(e[0] + e[2]) / 2, (e[1] + e[3]) / 2] for e in edge_data])
    tree = cKDTree(midpoints)

    for i in range(len(OSMList)):
        item = OSMList[i]
        lat, lon = item[0], item[1]

        # Query radius — generous to catch nearby edges
        search_r = threshold * 4
        nearby_idx = tree.query_ball_point([lat, lon], search_r)

        min_dist = 10000
        min_edge = -1

        for idx in nearby_idx:
            edgeid = edge_ids[idx]
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            n3 = item[2]
            n4 = item[3]

            lat1, lon1 = GPSMap.nodes[n1]
            lat2, lon2 = GPSMap.nodes[n2]
            lat3, lon3 = OSMMap.nodes[n3]
            lat4, lon4 = OSMMap.nodes[n4]

            nlat1, nlon1 = latlonNorm((lat2 - lat1, lon2 - lon1))
            nlat2, nlon2 = latlonNorm((lat4 - lat3, lon4 - lon3))

            dist = pointToLineDistanceLatLon((lat1, lon1), (lat2, lon2), (lat, lon))
            if dist < threshold and dist < min_dist:
                angle_dist = 1.0 - abs(nlat1 * nlat2 + nlon1 * nlon2)
                if angle_dist < 0.04:  # 15 degrees
                    min_edge = edgeid
                    min_dist = dist

        if min_edge != -1:
            edgeid = min_edge
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            lat1, lon1 = GPSMap.nodes[n1]
            lat2, lon2 = GPSMap.nodes[n2]
            result[i] = [edgeid, n1, n2,
                         distance((lat1, lon1), (lat, lon)),
                         distance((lat2, lon2), (lat, lon)),
                         lat, lon]

    return result


# ── TOPO 1-to-1 Matching ────────────────────────────────────────────────────

def TOPO121(topo_result, roadgraph):
    if not topo_result:
        return topo_result

    # Build spatial index
    points = np.array([[r[0], r[1]] for r in topo_result])
    tree = cKDTree(points)

    new_list = []
    for ind in range(len(topo_result)):
        lat, lon = topo_result[ind][0], topo_result[ind][1]
        r_lat = 0.00030
        candidate = tree.query_ball_point([lat, lon], r_lat)

        competitors = []
        gpsn1, gpsn2 = topo_result[ind][4], topo_result[ind][5]
        gpsd1, gpsd2 = topo_result[ind][6], topo_result[ind][7]

        for can_id in candidate:
            t_gpsn1, t_gpsn2 = topo_result[can_id][4], topo_result[can_id][5]
            t_gpsd1, t_gpsd2 = topo_result[can_id][6], topo_result[can_id][7]

            d = roadgraph.distanceBetweenTwoLocation(
                (gpsn1, gpsn2, gpsd1, gpsd2),
                (t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2),
                max_distance=0.00030)
            if d < 0.00020:
                competitors.append(can_id)

        new_list.append((topo_result[ind], ind, competitors))

    def get_key(item):
        return item[0][2]  # precision

    new_list = sorted(new_list, key=get_key)
    result = []
    mark = {}

    for ind in range(len(new_list) - 1, -1, -1):
        if new_list[ind][1] in mark:
            if new_list[ind][0][2] < 0.9:
                continue
        result.append(new_list[ind][0])
        for cc in new_list[ind][2]:
            mark[cc] = 1

    return result


def topoAvg(topo_result):
    p, r = 0, 0
    for item in topo_result:
        p += item[2]
        r += item[3]
    if len(topo_result) == 0:
        return 0, 0
    return p / len(topo_result), r / len(topo_result)


# ── Main TOPO Evaluation ────────────────────────────────────────────────────

def TOPOWithPairs(GPSMap, OSMMap, GPSList, OSMList,
                  step=0.00005, r=0.00300, threshold=0.00015,
                  outputfile="tmp.txt", one2oneMatching=True):
    i = 0
    precesion_sum = 0
    recall_sum = 0
    rrr = float(len(GPSList.keys())) / float(len(OSMList)) if len(OSMList) > 0 else 0
    returnResult = []

    for k, itemGPS in GPSList.items():
        itemOSM = OSMList[k]
        gpsn1, gpsn2, gpsd1, gpsd2 = itemGPS[1], itemGPS[2], itemGPS[3], itemGPS[4]
        osmn1, osmn2, osmd1, osmd2 = itemOSM[2], itemOSM[3], itemOSM[4], itemOSM[5]

        marbles = GPSMap.TOPOWalk(1, step=step, r=r, direction=False,
                                  newstyle=True, nid1=gpsn1, nid2=gpsn2,
                                  dist1=gpsd1, dist2=gpsd2)
        holes = OSMMap.TOPOWalk(1, step=step, r=r, direction=False,
                                newstyle=True, nid1=osmn1, nid2=osmn2,
                                dist1=osmd1, dist2=osmd2)
        holes_bidirection = OSMMap.TOPOWalk(1, step=step, r=r, direction=False,
                                            newstyle=True, nid1=osmn1, nid2=osmn2,
                                            dist1=osmd1, dist2=osmd2,
                                            bidirection=True)

        # Build KDTrees for spatial queries
        if len(marbles) == 0 or len(holes) == 0:
            continue

        marble_pts = np.array([[m[0], m[1]] for m in marbles])
        hole_pts = np.array([[h[0], h[1]] for h in holes])
        hole_bi_pts = np.array([[h[0], h[1]] for h in holes_bidirection])

        marble_tree = cKDTree(marble_pts)
        hole_bi_tree = cKDTree(hole_bi_pts)

        # ── Precision: match marbles → holes_bidirection ──
        matchedNum = 0
        bigraph = {}
        rr = threshold * 1.8

        for m_idx, marble in enumerate(marbles):
            possible = hole_bi_tree.query_ball_point([marble[0], marble[1]], rr)
            for hole_id in possible:
                hole = holes_bidirection[hole_id]
                ddd = distance(marble, hole)

                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))

                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0

                if ddd < threshold and angle_d < 0.29:
                    if marble in bigraph:
                        bigraph[marble].add(hole_id)
                    else:
                        bigraph[marble] = set([hole_id])
                    matchedNum += 1

        if one2oneMatching and bigraph:
            matches = _hopcroft_karp_matching(bigraph)
            matchedNum = len(matches) // 2

        precesion = float(matchedNum) / len(marbles)

        # ── Recall: match holes → marbles ──
        matchedNum = 0
        bigraph = {}

        for h_idx, hole in enumerate(holes):
            possible = marble_tree.query_ball_point([hole[0], hole[1]], rr)
            for marble_id in possible:
                marble = marbles[marble_id]
                ddd = distance(marble, hole)

                n1 = latlonNorm((marble[2], marble[3]))
                n2 = latlonNorm((hole[2], hole[3]))

                if marble[2] != marble[3] and hole[2] != hole[3]:
                    angle_d = 1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1])
                else:
                    angle_d = 0.0

                if ddd < threshold and angle_d < 0.29:
                    if hole in bigraph:
                        bigraph[hole].add(marble_id)
                    else:
                        bigraph[hole] = set([marble_id])
                    matchedNum += 1

        if one2oneMatching and bigraph:
            matches = _hopcroft_karp_matching(bigraph)
            matchedNum = len(matches) // 2

        recall = float(matchedNum) / len(holes)

        precesion_sum += precesion
        recall_sum += recall

        if outputfile:
            with open(outputfile, "a") as fout:
                fout.write(f"{i} {itemOSM[0]} {itemOSM[1]} {gpsn1} {gpsn2} "
                           f"Precesion {precesion} Recall {recall} "
                           f"Avg Precesion {precesion_sum / (i + 1)} "
                           f"Avg Recall {recall_sum / (i + 1)} \n")

        returnResult.append((itemOSM[0], itemOSM[1], precesion, recall,
                             gpsn1, gpsn2, gpsd1, gpsd2))
        i += 1

    new_topoResult = TOPO121(returnResult, GPSMap)
    p, r = topoAvg(new_topoResult)
    overall_recall = r * len(new_topoResult) / float(len(OSMList)) if len(OSMList) > 0 else 0

    if outputfile:
        try:
            with open(outputfile, "a") as fout:
                fout.write(f"precision={p} overall-recall={overall_recall}")
        except Exception:
            pass

    return p, overall_recall


# ── Public API ───────────────────────────────────────────────────────────────

def evaluate_topo(gt_graph_path, pred_graph_path, output_path,
                  matching_threshold=0.00010, interval=0.00005):
    """
    Run the full TOPO metric. Direct replacement for the subprocess call.
    Returns (precision, overall_recall) or None on failure.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Clear output file
    if os.path.exists(output_path):
        os.remove(output_path)

    map_gt = pickle.load(open(gt_graph_path, "rb"))
    map_prop = pickle.load(open(pred_graph_path, "rb"))

    graph_gt, min_lat_gt, max_lon_gt = create_graph_from_pickle(map_gt)
    graph_prop, min_lat_prop, max_lon_prop = create_graph_from_pickle(map_prop)

    min_lat = min(min_lat_gt, min_lat_prop)
    max_lon = max(max_lon_gt, max_lon_prop)

    region = [min_lat - 300 / 111111.0, LON_TOP_LEFT - 500 / 111111.0,
              LAT_TOP_LEFT + 300 / 111111.0, max_lon + 500 / 111111.0]
    graph_gt.region = region
    graph_prop.region = region

    losm = TOPOGenerateStartingPoints(graph_gt, region=region)
    lmap = TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold=matching_threshold)

    # Propagation distance
    r = 0.00300
    if LAT_TOP_LEFT - min_lat < 0.01000:
        r = 0.00150

    precision, overall_recall = TOPOWithPairs(
        graph_prop, graph_gt, lmap, losm,
        r=r, step=interval, threshold=matching_threshold,
        outputfile=output_path, one2oneMatching=True)

    return precision, overall_recall
