import sys
sys.path.append("../")

from pred_pano_labels import pred_pano_labels

print pred_pano_labels("4s6C3NR6YRvHCYKMM_00QQ", "../panos/", 16384, 8192, "../models/", num_threads=4, save_labeled_pano=True, verbose=True)
