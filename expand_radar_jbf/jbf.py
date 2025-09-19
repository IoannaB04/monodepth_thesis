import numpy as np


def jbf_method(radar_depth, image,
               f_u, f_v,
               LUT_s, LUT_r, MAX_SHIFT,
               w=1, h=2,
               creat_labels=False):

    # MAX_SHIFT = next((i for i, j in enumerate(LUT_s < 0.1) if j), None)

    # output variables
    ex_depth = radar_depth
    ex_conf_score = np.zeros(radar_depth.shape)

    Y, X = np.where(radar_depth > 0)  # positions of non-zero values
    D = radar_depth[Y, X]             # depth values of non-zero elements
    sort_index = D.argsort()          # sorted from min to max

    # Create the expanded depth and confidence score matrices with an additional dimension
    height, width, _ = image.shape
    expanded_depth = np.zeros((height, width, len(D)))
    expanded_conf_score = np.zeros((height, width, len(D)))

    # label array
    if creat_labels:
        label_array = np.zeros((height, width, 2))
        # 1st dimension --> radar point that the pixel belongs to
        # 2nd dimension --> 1 if that pixel corresponds to a radar point

        label_array[:, :, 0] += 1000 # labels start from 0, so I give a big number for the background

    # Iterate over each radar depth point
    for depth_idx, _index in enumerate(sort_index):
        p_x = X[_index]
        p_y = Y[_index]
        p_d = D[_index]
        p_i = image[p_y, p_x]

        ex_conf_score[p_y, p_x] = 1
        expanded_conf_score[p_y, p_x, depth_idx] = 1  # Initial confidence for the specific depth layer

        if creat_labels:
            label_array[p_y, p_x, 0] = depth_idx
            label_array[p_y, p_x, 1] = 1

        # Define the relative window for each radar point
        v = int((h * f_v) / p_d)
        u = int((w * f_u) / p_d)
        dv = int(np.min([MAX_SHIFT, int(v / 2)]))
        du = int(np.min([MAX_SHIFT, int(u / 2)]))

        for i in range(-du, du):
            for j in range(-dv, dv):
                q_y = np.max([0, np.min([height - 1, p_y + j])])
                q_x = np.max([0, np.min([width - 1, p_x + i])])
                q_i = image[q_y, q_x]

                if radar_depth[q_y, q_x]: continue

                d_x = np.abs(q_x - p_x)
                d_y = np.abs(q_y - p_y)

                d_r = np.abs(int(p_i[0]) - int(q_i[0]))
                d_g = np.abs(int(p_i[1]) - int(q_i[1]))
                d_b = np.abs(int(p_i[2]) - int(q_i[2]))
                d_i = np.sqrt(d_r**2 + d_g**2 + d_b**2)

                G_s = LUT_s[d_x] * LUT_s[d_y]
                G_i = LUT_r[int(d_i)]
                G_jbf = G_s * G_i

                if G_jbf > 0.05:
                    expanded_depth[q_y, q_x, depth_idx] = p_d
                    expanded_conf_score[q_y, q_x, depth_idx] = G_jbf

    # Update ex_depth and ex_conf_score based on max confidence score in all dimensions (radar points)
    for y in range(height):
        for x in range(width):
            if np.any(expanded_conf_score[y, x, :]):
                max_conf_idx = np.argmax(expanded_conf_score[y, x, :])
                max_conf_value = expanded_conf_score[y, x, max_conf_idx]

                if max_conf_value > 0:
                    ex_depth[y, x] = expanded_depth[y, x, max_conf_idx] # Update ex_depth with the depth to the max confidence
                    ex_conf_score[y, x] = max_conf_value                # Update ex_conf_score with the maximum confidence
                    if creat_labels:
                        label_array[y, x, 0] = max_conf_idx

    # visualize_label_array(label_array, image, ex_depth)

    if creat_labels:
        return ex_depth, ex_conf_score, label_array
    else:
        return ex_depth, ex_conf_score


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_label_array(label_array, image, ex_depth, point_size_mapping=None):
    if point_size_mapping is None:
        point_size_mapping = {'default': 0.01}  # Default point size mapping if not provided

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # First Dimension Visualization (Unique Integers with Different Colors)
    unique_labels = np.unique(label_array[:, :, 0])
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    colored_label_array = np.vectorize(label_map.get)(label_array[:, :, 0])
    first_dim_img = axs[0].imshow(colored_label_array, cmap='tab20')
    axs[0].set_title("First Dimension (Unique Colors for Each Integer)")
    axs[0].axis('off')

    # Add color bar for the first dimension
    cbar = fig.colorbar(first_dim_img, ax=axs[0], orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(unique_labels)))
    cbar.set_ticklabels(unique_labels)
    cbar.set_label("Unique Integer Values")

    # Second Dimension Visualization (Overlay of radar points on image)
    axs[1].imshow(image)
    white_pixel_count = np.sum(label_array[:, :, 1] == 1)

    # Extract non-zero positions and corresponding labels from label_array
    y, x = np.where(label_array[:, :, 1] == 1)
    label_values = label_array[y, x, 0]  # Use the first dimension for color coding
    label_colors = [label_map[val] for val in label_values]

    # Overlay scatter plot without color bar
    axs[1].scatter(x, y, c=label_colors, cmap='tab20', s=5)
    axs[1].set_title(f"Radar Points Overlay on Image\nPixels: {white_pixel_count}")
    axs[1].axis('off')

    # Third Dimension: ex_depth Visualization
    axs[2].imshow(image)  # Count radar points

    # Non-zero depth points
    non_zero_indices = np.nonzero(ex_depth)
    non_zero_depths = ex_depth[non_zero_indices]
    scatter = axs[2].scatter(non_zero_indices[1], non_zero_indices[0], c=non_zero_depths, cmap='jet',
                             s=point_size_mapping.get('default', 5))

    # Set color normalization for depth values
    if non_zero_depths.size > 0:
        vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
    else:
        vmin, vmax = 0, 100
    scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))

    # Color bar for ex_depth
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Depth (meters)')
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))

    axs[2].set_title("ex_depth Overlay on Image")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

