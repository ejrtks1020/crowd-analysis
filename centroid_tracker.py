import math

class EuclideanDistTracker:
    def __init__(self, disappear_tolerance=20):
        self.center_points = {}
        self.id_count = 0
        self.disappear_tolerance = disappear_tolerance

    def update(self, objects_rect):
        current_id_set = []
        objects_bbs_ids = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            area = (x2 - x1) * (y2 - y1)
            dist_threshold = area * 0.005
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            same_object_detected = False
            if self.center_points:
                dists = [ (id, math.hypot(center_x - pt["xy"][0], center_y - pt["xy"][1])) for id, pt in self.center_points.items() ]
                dists = sorted(dists, key=lambda x : x[1])

                for id, dist in dists:
                    if id in current_id_set:
                        continue
                    # print(f"dist threshold : {dist_threshold}, dist : {dist}")
                    if dist < 500:
                        self.center_points[id] = {"xy":(center_x, center_y), "disappear":0}
                        objects_bbs_ids.append([x1,y1,x2,y2,id])
                        same_object_detected = True
                        current_id_set.append(id)  
                    break

                # min_dist_id, min_dist = dists[0][0], dists[0][1]
                # # print(min_dist)
                # if min_dist < 500:
                #     self.center_points[min_dist_id] = {"xy":(center_x, center_y), "disappear":0}
                #     objects_bbs_ids.append([x1,y1,x2,y2,min_dist_id])
                #     same_object_detected = True

            if not same_object_detected:
                self.center_points[self.id_count] = {"xy":(center_x, center_y), "disappear":0}
                objects_bbs_ids.append([x1,y1,x2,y2, self.id_count])
                current_id_set.append(self.id_count)
                self.id_count += 1
        
        current_ids = [bb_id[-1] for bb_id in objects_bbs_ids]
        # self.center_points = {id : pt for id, pt in self.center_points.items() if id in current_ids}

        new_center_points = {}
        for id, xy_disappear in self.center_points.items():
            if id in current_ids:
                new_center_points[id] = xy_disappear
            elif xy_disappear["disappear"] <= self.disappear_tolerance:
                new_center_points[id] = {"xy" : xy_disappear["xy"], "disappear" : xy_disappear["disappear"] + 1}
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
