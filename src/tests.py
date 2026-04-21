# from dataloader import GenDEBRIS
# import os
# import sys 
# import numpy as np
# import matplotlib.pyplot as plt
# from utils.assets import labels_agg
# import matplotlib.patches as mpatches
# from utils.assets import labels_agg, color_mapping, S2_RGB_INDICES
# from matplotlib.colors import ListedColormap, BoundaryNorm

# DATA_PATH = os.path.join(os.path.dirname(__file__),'..', 'data', 'MARIDA')
# CLASS_COLORS = [color_mapping[label] for label in labels_agg]
# CLASS_CMAP = ListedColormap(CLASS_COLORS)
# CLASS_NORM = BoundaryNorm(boundaries=np.arange(-0.5, len(labels_agg)), ncolors=len(labels_agg))
      
# if __name__ == "__main__":
#     dataset = GenDEBRIS(mode="train", path= DATA_PATH, agg_to_water=True)
#     print(dataset.X[0]) # this is the raw images 
#     print(dataset.y[0]) # this is the raw masks
#     img, target = dataset[221] # when you call __getitem__ it processes and is ready for the model
#                              # it reorders the index from (C,H,W) to (H,W,C)
#                              # does transformations and normalization
#                              # Imputes NaNs with band means to avoid NaN values 
                             
#     rgb = img[:, :, S2_RGB_INDICES] # takes the R,G,B from the image to visualize an image
#       # Stretch to [0, 1] for display

#     #the following for loop is AI-generated
#     for ch in range(3):
#         p2, p98 = np.percentile(rgb[:, :, ch], [2, 98])
#         rgb[:, :, ch] = np.clip((rgb[:, :, ch] - p2) / (p98 - p2 + 1e-8), 0, 1)
        
    
#     #Be able to print the pictures correctly and beautifully. TODO before modeling. 
    
#     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#     # ax1.imshow(rgb)
#     # ax1.set_title("RGB")
#     # ax1.axis('off')

#     # present_labels = np.unique(target)
#     # present_labels = present_labels[present_labels >= 0]

#     # cmap = plt.cm.get_cmap('tab20', 11)
#     # im = ax2.imshow(target, cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation='nearest')
#     # ax2.axis('off')

#     # present_labels = np.unique(target)
#     # present_labels = present_labels[present_labels >= 0]
#     # patches = [mpatches.Patch(color=CLASS_COLORS[i], label=labels_agg[i]) for i in present_labels]
#     # ax2.legend(handles=patches, fontsize=7, loc='lower right')
#     # plt.show()