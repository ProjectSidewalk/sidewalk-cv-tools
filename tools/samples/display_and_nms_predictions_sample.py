# Written by Kavi Dey August 2019

import sys
sys.path.append("../")

from cv_tools import get_pano_labels

if __name__ == "__main__":
    print(get_pano_labels("1a1UlhadSS_3dNtc5oI10Q","panos/", save_labeled_pano=True, verbose=True))
    print(get_pano_labels("4s6C3NR6YRvHCYKMM_00QQ","panos/", save_labeled_pano=True, verbose=True))
