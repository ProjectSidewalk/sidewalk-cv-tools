import sys
sys.path.append("../")

from cv_tools import batch_save_pano_labels

if __name__ == "__main__":
    batch_save_pano_labels(["1a1UlhadSS_3dNtc5oI10Q", "4s6C3NR6YRvHCYKMM_00QQ"], "panos/", "models/", num_threads=4, verbose=True)
