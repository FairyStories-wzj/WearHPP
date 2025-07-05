from data_loader.skeleton import Skeleton
class Xrf2Skeleton(Skeleton):
    def __init__(self):
        parents = [-1,0,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
        joints_left = [2,3,4,9,10,11,22,23,24]
        joints_right = [5,6,7,12,13,14,19,20,21]
        super(Xrf2Skeleton, self).__init__(parents, joints_left, joints_right)
        removed_joints = [15,16,17,18,19,20,21,22,23,24]
        self.remove_joints(removed_joints)
