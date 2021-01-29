'''
CUSTOM PAINT-BY-NUMBERS GENERATOR

Jason Natale
'''

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import copy
import math


# get color distance of L*a*b colors, just doin euclidean
def get_color_dist(color1, color2):
    col1 = color1.tolist()
    col2 = color2.tolist()
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(col1, col2)]))


# Process pixel at row i, column j
def process_pixel(i, j):
    global island_cnt, min_brush_size
    # Found new contiguous segment of color - aka an "island"
    if img_info[i][j] == 0:
        print('New Island at ({}, {})'.format(i, j))
        img_info[i][j] = island_cnt
        island_color = island_img[i][j]
        pixel_queue = []
        # attempt to expand right, left, up, down
        pixel_queue.append((i, j+1))
        pixel_queue.append((i, j-1))
        pixel_queue.append((i-1, j))
        pixel_queue.append((i+1, j))
        info = {'size': 1, 'color': island_color, 'start_pt': (i, j), 'num': island_cnt, 'pts': [(i,j)]}
        nearest_color_info = {'min_diff': None, 'color': None, 'island_num': None}
        # start processing queue
        while len(pixel_queue) > 0:
            pixel = pixel_queue.pop()
            expand_island(pixel[0], pixel[1], info, pixel_queue, nearest_color_info)
        # when island too small need to consolidate with neighbor of nearest color and re-run
        if info['size'] < min_brush_size:
            # this if stmt will always be true, but adding it for safety
            if nearest_color_info['color'] is not None:
                for (y,x) in info['pts']:
                    island_img[y][x] = nearest_color_info['color']
                    new_island_num = nearest_color_info['island_num']
                    img_info[y][x] = new_island_num if new_island_num is not None else 0
            process_pixel(i, j)
        else:
            # post-processing
            island_info[island_cnt] = info
            island_cnt += 1


# Attempt to expand island to point at row i, column j
def expand_island(i, j, info, pixel_queue, nearest_color_info):
    global row_num, col_num, min_brush_size
    valid_pt = i > -1 and j > -1 and i < row_num and j < col_num
    # island expands when reaching a pixel of the same color which isn't yet marked as belonging to the island's num
    if valid_pt and img_info[i][j] != info['num'] and np.array_equal(island_img[i][j], info['color']):# img_info[i][j] == 0
        img_info[i][j] = info['num']
        island_img[i][j] = info['color']
        info['size'] += 1
        info['pts'].append((i, j))
        # attempt to expand right, left, up, down
        pixel_queue.append((i, j+1))
        pixel_queue.append((i, j-1))
        pixel_queue.append((i-1, j))
        pixel_queue.append((i+1, j))
    # nearest color logic for those colors which are not the same, added second constraint to save time
    elif valid_pt and info['size'] < min_brush_size and not np.array_equal(island_img[i][j], info['color']):
        diff = get_color_dist(island_img[i][j], info['color'])
        if nearest_color_info['min_diff'] is None or nearest_color_info['min_diff'] > diff:
            nearest_color_info['min_diff'] = diff
            nearest_color_info['color'] = island_img[i][j]
            if img_info[i][j] is not None:
                nearest_color_info['island_num'] = img_info[i][j]


# Search for first pixel of an island, and trace entire island
def trace_island(i, j):
    # new island to trace
    vals_of_interest = []
    pts_of_interest = []
    if (i, j) in pts_of_interest:
        print('at point of interest... interesting value of {}'.format(img_info[i][j]))
    if img_info[i][j] not in islands_traced:
        if (i, j) in pts_of_interest:
            print('Tracing island at ({}, {})'.format(i, j))
        if img_info[i][j] in vals_of_interest:
            print('{} was spotted first at {}, {}'.format(img_info[i][j], i, j))
        islands_traced.append(img_info[i][j])
        paint_by_number_canvas[i][j] = np.array([0, 0, 0])
        path = [(i, j)]
        next_pt = None
        search_dir = 1
        island_num = img_info[i][j]
        trace_next_pt(island_num, path, search_dir)
        # as long as path doesn't cycle back to start, keep going
        while path[0] != path[-1]:
            search_dir = trace_next_pt(island_num, path, search_dir)

# Find next point in an island path by cycling around existing pt
# handles a single cycle, include directional to start cycle
# right = 1; down-right = 2 and so on clockwise...
def trace_next_pt(island_num, path, search_dir):
    global row_num, col_num
    (row, col) = path[-1]
    next_pt = None
    potential_pts = []
    potential_pts.append(((row - 1, col + 1), False if row == 0 or col+1 == col_num else True, 0))
    potential_pts.append(((row, col + 1), False if col+1 == col_num else True, 1))
    potential_pts.append(((row + 1, col + 1), False if row+1 == row_num or col+1 == col_num else True, 2))
    potential_pts.append(((row + 1, col), False if row+1 == row_num else True, 3))
    potential_pts.append(((row + 1, col - 1), False if col == 0 or row+1 == row_num else True, 4))
    potential_pts.append(((row, col - 1), False if col == 0 else True, 5))
    potential_pts.append(((row - 1, col - 1), False if row == 0 or col == 0 else True, 6))
    potential_pts.append(((row - 1, col), False if row == 0 else True, 7))
    #print('coming from pt {}'.format(path[-1]))
    # convert points list into new list in order from starting position
    val = search_dir
    #print('search dir is {}'.format(search_dir))
    ordered_adj_pts = [potential_pts[val]]
    val = (val + 1) % 8
    while val != search_dir:
        ordered_adj_pts.append(potential_pts[val])
        val = (val + 1) % 8

    #print(ordered_adj_pts)
    for (y,x), valid, idx in ordered_adj_pts:
        #if (y, x) == (8, 75):
            #print('{} : {} : {} : {}'.format(valid, next_pt, img_info[y][x], island_num))
        if not valid or next_pt is not None:
            continue
        elif img_info[y][x] == island_num:
            paint_by_number_canvas[y][x] = np.array([0, 0, 0])
            next_pt = (y, x)
            #print('Next pt is {}'.format(next_pt))
            path.append(next_pt)
            # next directional is always 90deg counter clockwise to the direction we just moved
            search_dir = (idx + 6) % 8
            return search_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-c", "--max-colors", required = True, type = int, help = "Max number of distinct colors")
    ap.add_argument("-s", "--min-brush-size", required = True, type = int)
    ap.add_argument("-n", "--name-suffix", required = True, help = "File names suffix")
    ap.add_argument("-f", "--scale-factor", required = False, type = int, default = 1, help = "Scale multiplier between input image and paint-by-number canvas")
    args = vars(ap.parse_args())

    min_brush_size = args["min_brush_size"]
    max_colors = args["max_colors"]
    name_suffix = args["name_suffix"]

    # load the image and grab its width and height
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]

    print('Processing Image of size ({}, {})'.format(h, w))
    print('Quantizing Image...')

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = max_colors)
    labels = clt.fit_predict(image)
    quantized = clt.cluster_centers_.astype("uint8")[labels]
    quantized_img = quantized.reshape((h, w, 3))
    # bring image back to BGR color scheme
    quantized_img = cv2.cvtColor(quantized_img, cv2.COLOR_LAB2BGR)
    orig_quant = np.copy(quantized_img)

    # quant and islands_info are both of the form row x column
    col_num = len(quantized_img[0])
    row_num = len(quantized_img)
    img_row = [0 for _ in range(col_num)]
    img_info = [copy.deepcopy(img_row) for _ in range(row_num)]
    island_cnt = 1
    island_img = copy.deepcopy(quantized_img)
    island_info = {}
    white_canvas = copy.deepcopy(quantized_img)

    print('Detecting and Defining all contiguous color regions (islands)...')
    # First loop over all pixels detecting and consolidating "islands" of color,
    # while ensuring that "islands" smaller than min-brush-size are merged with larger ones.
    # Also, create the white canvas to be used later.
    for i in range(row_num):
        for j in range(col_num):
            white_canvas[i][j] = np.array([255, 255, 255])
            process_pixel(i, j)

    color_sheet = copy.deepcopy(white_canvas)
    paint_by_number_canvas = copy.deepcopy(white_canvas)

    print('Tracing Islands...')
    islands_traced = []
    # Next, iterate over all pixels again to trace "island" outlines
    for i in range(row_num):
        for j in range(col_num):
            trace_island(i, j)

    # Find all colors use and add to color_sheet
    colors = []
    for island_id in island_info.keys():
        info = island_info[island_id]
        curr_color = (info['color'][0], info['color'][1], info['color'][2])
        if curr_color not in colors:
            colors.append(curr_color)
            if len(colors) == max_colors:
                break

    print('\nCompleted!')
    print('{} Distinct BGR Colors'.format(len(colors)))

    original_height, original_width = white_canvas.shape[:2]
    rows_of_color = (int)(math.ceil(len(colors) / 10))
    inc_width = (int)(original_width / 10)
    inc_height = (int)(original_height / rows_of_color)
    for x, color in enumerate(colors):
        width = (int)(x % 10 * inc_width)
        height = (int)(math.floor(x / 10) * inc_height if x!= 0 else 0)
        top_left = (int(width), int(height))
        bottom_right = (int(width + inc_width), int(height + inc_height))
        # make input color tuple of type int
        input_color = ((int)(color[0]), (int)(color[1]), (int)(color[2]))
        print('Color {} is {}'.format(x, input_color))
        color_sheet = cv2.rectangle(color_sheet, top_left, bottom_right, input_color, -1)

    preview = island_img

    # Post-Process Paint by Numbers
    factor = args["scale_factor"]
    paint_by_number_canvas = cv2.resize(paint_by_number_canvas, (int(original_width*factor), int(original_height*factor)), interpolation=cv2.INTER_CUBIC )
    paint_by_number = paint_by_number_canvas

    # Save
    cv2.imwrite('color_sheet_{}.jpg'.format(name_suffix), color_sheet)
    cv2.imwrite('paint_by_number_{}.jpg'.format(name_suffix), paint_by_number)
    cv2.imwrite('preview_{}.jpg'.format(name_suffix), preview)

    # Optional image display
    #cv2.imshow("traced", paint_by_number)
    #cv2.imshow("color_sheet", color_sheet)
    #cv2.imshow("preview", preview)
    #cv2.waitKey(0)
