import pandas as pd 
import numpy as np

from ethome.io import XY_IDS

from itertools import product

#Shift decorator
#If a feature creation function has this applied then the features are
#automatically shifted/differences/had distribution stats computed and
#concatenated with the rest of the table

#The decorator maker, so we can provide arguments
def augment_features(window_size = 5, n_shifts = 3, mode = 'shift'):
    #The decorator
    def decorator(feature_function):
        #What is called instead of the actual function, assumes the feature making
        #function returns the names of the columns just made
        def wrapper(*args, **kwargs):
            #Compute the features
            #print(args, kwargs, args[0])
            if 'mode' in kwargs:
                mode = kwargs['mode']
            if 'n_shifts' in kwargs:
                n_shifts = kwargs['n_shifts']

            old_cols = set(args[0].columns)
            df = feature_function(*args, **kwargs)
            if n_shifts == 0: return df
            new_cols = set(df.columns)
            added_cols = list(new_cols.difference(old_cols))
            #Shift the features just made
            shifted_data = []
            if mode == 'distr':
                window_sizes = [1, 5, 10]
                for ws in window_sizes:
                    data = np.dstack([np.array(df[added_cols].shift(p)) for p in range(-ws, ws+1)])
                    min_data = pd.DataFrame(np.min(data, axis = 2), columns = [f'{cn}_min_pm_{ws}' for cn in added_cols])
                    max_data = pd.DataFrame(np.max(data, axis = 2), columns = [f'{cn}_max_pm_{ws}' for cn in added_cols])
                    std_data = pd.DataFrame(np.std(data, axis = 2), columns = [f'{cn}_std_pm_{ws}' for cn in added_cols])
                    mean_data = pd.DataFrame(np.mean(data, axis = 2), columns = [f'{cn}_mean_pm_{ws}' for cn in added_cols])
                    shifted_data += [min_data, max_data, std_data, mean_data]
            else:
                periods = [-(i+1)*window_size for i in range(n_shifts)] + \
                        [(i+1)*window_size for i in range(n_shifts)]
                #Rename all column names
                for p in periods:
                    if mode == 'shift':
                        s_df = df[added_cols].shift(p)
                    elif mode == 'diff':
                        s_df = df[added_cols].diff(p)
                    s_df = s_df.rename(columns = {k:f'{k}_shifted_{p}' for k in added_cols})
                    shifted_data.append(s_df)
            #Combine with current table
            #TODO
            # Figure out why a reset_index is needed here... seems to cause issues downstream
            # df has a funny index or column structure?
            #df = pd.concat([df.reset_index(drop = True)] + shifted_data, axis = 1)
            df = pd.concat([df] + shifted_data, axis = 1)
            return df
        return wrapper
    return decorator

from pandas.api.types import is_numeric_dtype

def boiler_plate(features_df):
    reversemap = None

    to_drop = ['Unnamed: 0']
    for col in to_drop:
        if col in features_df.columns:
            features_df = features_df.drop(columns = col)
    #Impute nas
    for col in features_df:
        if is_numeric_dtype(features_df[col]):
            features_df[col] = features_df[col].fillna(features_df[col].mean())
    return features_df, reversemap

@augment_features()
def _compute_centroid(df, name, animal_setup, body_parts = None, n_shifts = 3, mode = 'shift'):
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']

    if body_parts is None:
        body_parts = bodypart_ids

    df = df.copy()
    for mouse_id in mouse_ids:
        part_names_x = [f'{mouse_id}_x_{i}' for i in body_parts]
        part_names_y = [f'{mouse_id}_y_{i}' for i in body_parts]
        df[f'centroid_{name}_{mouse_id}_x'] = np.mean(df[part_names_x], axis = 1)
        df[f'centroid_{name}_{mouse_id}_y'] = np.mean(df[part_names_y], axis = 1)
    return df

@augment_features()
def _compute_abs_angle(df, name, animal_setup, bps, centroid = True, n_shifts = 3, mode = 'shift'):
    mouse_ids = animal_setup['mouse_ids']
    df = df.copy()
    if len(bps) != 2:
        raise ValueError('Abs angle only works between 2 bodyparts, too many or too few specified')
    for mouse_id in mouse_ids:
        if centroid:
            diff_x = df[f'{bps[0]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y = df[f'{bps[0]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
        else:
            diff_x = df[f'{mouse_id}_x_{bps[0]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y = df[f'{mouse_id}_y_{bps[0]}'] - df[f'{mouse_id}_y_{bps[1]}']
        df[f'angle_{name}_{mouse_id}'] = np.arctan2(diff_y,diff_x)  
    return df

@augment_features()
def _compute_rel_angle(df, name, animal_setup, bps, centroid = False, n_shifts = 3, mode = 'shift'):
    mouse_ids = animal_setup['mouse_ids']    
    df = df.copy()
    if len(bps) != 3:
        raise ValueError('too many body parts to compute an absolute angle. Only works for 2')
    for mouse_id in mouse_ids:
        if centroid:
            diff_x1 = df[f'{bps[0]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y1 = df[f'{bps[0]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
            diff_x2 = df[f'{bps[2]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y2 = df[f'{bps[2]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
        else:
            diff_x1 = df[f'{mouse_id}_x_{bps[0]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y1 = df[f'{mouse_id}_y_{bps[0]}'] - df[f'{mouse_id}_y_{bps[1]}']
            diff_x2 = df[f'{mouse_id}_x_{bps[2]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y2 = df[f'{mouse_id}_y_{bps[2]}'] - df[f'{mouse_id}_y_{bps[1]}']

        diff1 = np.vstack((diff_x1, diff_y1)).T
        diff2 = np.vstack((diff_x2, diff_y2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_{name}_{mouse_id}'] = np.arccos(cosine_angle)
    return df

@augment_features()
def _compute_ellipsoid(df, animal_setup, n_shifts = 3, mode = 'shift'):
    
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']

    df = df.copy()
    #Perform SVD
    colnames = ['_'.join([a[0], a[2], a[1]]) for a in product(mouse_ids, bodypart_ids, XY_IDS)]
    data = np.array(df[colnames]).reshape(-1, 2, 7, 2)
    mean_data = np.transpose(np.tile(np.nanmean(data, axis = 2), (7,1,1,1)), (1,2,0,3))
    svd_data = np.nan_to_num(data-mean_data) 
    svals = np.linalg.svd(svd_data, compute_uv = False)
    #Not technically correct, but not sure if the square of the singular values is exaclty
    #what we want either. This keeps the scale roughly the same as the distances involved
    evals = svals
    # evals = (svals*svals)/6
    for idx,m_id in enumerate(mouse_ids):
        df[f'ellipse_major_{m_id}'] = evals[:,idx,0]
        df[f'ellipse_minor_{m_id}'] = evals[:,idx,1]
        ## ratio of major and minor
        df[f'ellipse_ratio_{m_id}'] = df[f'ellipse_minor_{m_id}']/df[f'ellipse_major_{m_id}']
        ## area of ellipse
        df[f'ellipse_area_{m_id}'] = df[f'ellipse_minor_{m_id}']*df[f'ellipse_major_{m_id}']

    ## ratio of areas of ellipses of the mice
    df[f'ellipse_area_ratio'] = df[f'ellipse_area_{mouse_ids[0]}']/df[f'ellipse_area_{mouse_ids[1]}']

    return df

#Recall framerate is 30 fps
def _compute_kinematics(df, names, animal_setup, window_size = 5, n_shifts = 3):
    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']
    df = df.copy()
    for mouse_id in mouse_ids:
        for name in names:
            ## Speed of centroids
            dx = df[f'centroid_{name}_{mouse_id}_x'].diff(window_size)
            dy = df[f'centroid_{name}_{mouse_id}_y'].diff(window_size)
            df[f'centroid_{name}_{mouse_id}_speed'] = np.sqrt(dx**2 + dy**2)
            #colnames.append(f'centroid_{name}_{mouse_id}_speed')
            ## Acceleration of centroids
            ddx = dx.diff(window_size)
            ddy = dy.diff(window_size)
            df[f'centroid_{name}_{mouse_id}_accel_x'] = ddx/(window_size**2)
            df[f'centroid_{name}_{mouse_id}_accel_y'] = ddy/(window_size**2)
    return df

@augment_features()
def _compute_relative_body_motions(df, animal_setup, window_size = 3, n_shifts = 3, mode = 'shift'):

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']
    
    #Compute vector connecting two centroids
    dx = df[f'centroid_all_{mouse_ids[0]}_x'] - df[f'centroid_all_{mouse_ids[1]}_x']
    dy = df[f'centroid_all_{mouse_ids[0]}_y'] - df[f'centroid_all_{mouse_ids[1]}_y']
    dm = np.sqrt(dx**2 + dy**2)
    df['distance_main_centroid'] = dm

    #Compute velocity of mouse centroids
    for m_id in mouse_ids:
        vx = df[f'centroid_all_{m_id}_x'].diff(window_size)/window_size
        vy = df[f'centroid_all_{m_id}_y'].diff(window_size)/window_size
        v_tangent = (dx*vx + dy*vy)/dm
        v_perp_x = vx - dx*v_tangent/dm
        v_perp_y = vy - dy*v_tangent/dm
        v_perp = np.sqrt(v_perp_x**2 + v_perp_y**2)
        df[f'relative_vel_tanget_{m_id}'] = v_tangent
        df[f'relative_vel_perp_{m_id}'] = v_perp

        ## relative distance scaled
        #Distance between main centroids, divided by length of major axis of each mouse
        df[f'scaled_main_centroid_distance_by_ellipse_major_{m_id}'] = dm/df[f'ellipse_major_{m_id}']

    return df

@augment_features()
def _compute_relative_body_angles(df, animal_setup, n_shifts = 3, mode = 'shift'):

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']

    for idx, m_id in enumerate(mouse_ids):
        #Compute vector connecting two centroids
        dx1 = df[f'centroid_all_{mouse_ids[1-idx]}_x'] - df[f'centroid_all_{mouse_ids[idx]}_x']
        dy1 = df[f'centroid_all_{mouse_ids[1-idx]}_y'] - df[f'centroid_all_{mouse_ids[idx]}_y']

        #Relative angle between body of mouse and line connecting two centroids
        dx2 = df[f'centroid_head_{m_id}_x'] - df[f'centroid_body_{m_id}_x']
        dy2 = df[f'centroid_head_{m_id}_y'] - df[f'centroid_body_{m_id}_y']            
        diff1 = np.vstack((dx1, dy1)).T
        diff2 = np.vstack((dx2, dy2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_head_body_centroid_{m_id}'] = np.arccos(cosine_angle)

        #Angle between head orientation of one mouse and line connecting two centroids
        dx2 = df[f'{m_id}_x_nose'] - df[f'{m_id}_x_neck']
        dy2 = df[f'{m_id}_x_nose'] - df[f'{m_id}_y_neck']            
        diff1 = np.vstack((dx1, dy1)).T
        diff2 = np.vstack((dx2, dy2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_head_centroid_{m_id}'] = np.arccos(cosine_angle)

        #Just threshold on if the angle is less than pi/4 radians
        df[f'{mouse_ids[1-idx]}_in_view_of_{mouse_ids[idx]}'] = (cosine_angle > 1/np.sqrt(2)).astype(float)

    return df
    
@augment_features()
def _compute_iou(df, animal_setup, n_shifts = 3, mode = 'shift'):

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']

    mins = {}
    maxs = {}

    for m_id in mouse_ids:
        for xy in XY_IDS:
            colnames = ['_'.join([m_id, xy, bp]) for bp in bodypart_ids]
            mins['_'.join([m_id, xy])] = np.min(df[colnames], axis = 1)
            maxs['_'.join([m_id, xy])] = np.max(df[colnames], axis = 1)

    dx = np.minimum(maxs[f'{mouse_ids[0]}_x'], maxs[f'{mouse_ids[1]}_x']) - np.maximum(mins[f'{mouse_ids[0]}_x'], mins[f'{mouse_ids[1]}_x'])
    dy = np.minimum(maxs[f'{mouse_ids[0]}_y'], maxs[f'{mouse_ids[1]}_y']) - np.maximum(mins[f'{mouse_ids[0]}_y'], mins[f'{mouse_ids[1]}_y'])
    dx = np.maximum(0, dx)
    dy = np.maximum(0, dy)

    bb1_area = (maxs[f'{mouse_ids[0]}_x'] - mins[f'{mouse_ids[0]}_x'])*(maxs[f'{mouse_ids[0]}_y'] - mins[f'{mouse_ids[0]}_y'])
    bb2_area = (maxs[f'{mouse_ids[1]}_x'] - mins[f'{mouse_ids[1]}_x'])*(maxs[f'{mouse_ids[1]}_y'] - mins[f'{mouse_ids[1]}_y'])
    intersection_area = dx*dy
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    df['iou'] = iou
    return df

##Distance from centroid of the mouse to the closest vertical edge
## and closest horizontal edge
##Distance to the closest edge
#These depend on the video you're applying it to...
#Which can change from video to video, train to test, etc. So perhaps not useful
@augment_features()
def _compute_cage_distances(features_df, animal_setup, n_shifts = 3, mode = 'shift'): # pragma: no cover

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']
    
    for m_id in mouse_ids:
        features_df[f'centroid_all_{m_id}_x_inverted'] = WIDTH - features_df[f'centroid_all_{m_id}_x']
        features_df[f'centroid_all_{m_id}_y_inverted'] = HEIGHT - features_df[f'centroid_all_{m_id}_y']
        features_df[f'{m_id}_closest_x'] = np.min(features_df[[f'centroid_all_{m_id}_x_inverted', f'centroid_all_{m_id}_x']], axis = 1)
        features_df[f'{m_id}_closest_y'] = np.min(features_df[[f'centroid_all_{m_id}_y_inverted', f'centroid_all_{m_id}_y']], axis = 1)
        features_df[f'{m_id}_closest'] = np.min(features_df[[f'{m_id}_closest_x', f'{m_id}_closest_y']], axis = 1)
        features_df = features_df.drop(columns = [f'centroid_all_{m_id}_x_inverted', f'centroid_all_{m_id}_y_inverted'])
    return features_df

def make_features_distances(df, animal_setup):

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']
    print(colnames)

    features_df = df.copy()

    ##Make the distance features
    for i, bp1 in enumerate(bodypart_ids):
        for j, bp2 in enumerate(bodypart_ids):
            if i < j:
                for mouse_id in mouse_ids:
                    #We can compute the intra-mouse difference
                    f1x = '_'.join([mouse_id, 'x', bp1])
                    f2x = '_'.join([mouse_id, 'x', bp2])
                    f1y = '_'.join([mouse_id, 'y', bp1])
                    f2y = '_'.join([mouse_id, 'y', bp2])
                    f_new = '_'.join([mouse_id, 'dist', bp1, bp2])
                    features_df[f_new] = \
                        np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                (features_df[f1y] - features_df[f2y])**2)
            #Inter-mouse difference
            f1x = '_'.join([mouse_ids[0], 'x', bp1])
            f2x = '_'.join([mouse_ids[1], 'x', bp2])
            f1y = '_'.join([mouse_ids[0], 'y', bp1])
            f2y = '_'.join([mouse_ids[1], 'y', bp2])
            f_new = '_'.join(['M0_M1', 'dist', bp1, bp2])
            features_df[f_new] = \
                        np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                (features_df[f1y] - features_df[f2y])**2)

    #Remove base features
    features_df = features_df.drop(columns = colnames)

    ##Clean up seq_id columns
    features_df, _ = boiler_plate(features_df)

    return features_df

def make_features_mars(df, animal_setup, n_shifts = 3, mode = 'shift'):

    features_df = df.copy()

    #######################
    ## Position features ##
    #######################
    features_df = _compute_centroid(features_df, 'all', animal_setup, n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'head', animal_setup, ['nose', 'leftear', 'rightear', 'neck'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'hips', animal_setup, ['lefthip', 'tail', 'righthip'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'body', animal_setup, ['neck', 'lefthip', 'righthip', 'tail'], n_shifts = n_shifts, mode = mode)

    #This is too specific to the particular cage setup, and requires knowing the cage (or image) dimensions, 
    #so we'll remove it.
    #features_df = _compute_cage_distances(features_df, n_shifts = n_shifts, mode = mode)

    #####################
    #Appearance features#
    #####################

    ## absolute orientation of mice
    features_df = _compute_abs_angle(features_df, 'head_hips', animal_setup, ['centroid_head', 'centroid_hips'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'head_nose', animal_setup, ['neck', 'nose'], centroid = False, n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'tail_neck', animal_setup, ['tail', 'neck'], centroid = False, n_shifts = n_shifts, mode = mode)
    ## relative orientation of mice
    features_df = _compute_rel_angle(features_df, 'leftear_neck_rightear', animal_setup, ['leftear', 'neck', 'rightear'], n_shifts = n_shifts, mode = mode)
    ## major axis len, minor axis len of ellipse fit to mouses body
    features_df = _compute_ellipsoid(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    #####################
    #Locomotion features#
    #####################

    features_df = _compute_kinematics(features_df, ['all', 'head', 'hips', 'body'], animal_setup)

    #################
    #Social features#
    #################

    features_df = _compute_relative_body_motions(features_df, animal_setup, n_shifts = n_shifts, mode = mode)
    features_df = _compute_relative_body_angles(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    #Intersection of union of bounding boxes of two mice
    features_df = _compute_iou(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    ## distance between all pairs of keypoints of each mouse
    features_df = make_features_distances(features_df, animal_setup)

    return features_df

def make_features_mars_distr(df, animal_setup):
    return make_features_mars(df, animal_setup, n_shifts = 3, mode = 'distr')

def make_features_mars_reduced(df, animal_setup, n_shifts = 2, mode = 'diff'):

    features_df = df.copy()

    #######################
    ## Position features ##
    #######################
    features_df = _compute_centroid(features_df, 'all', animal_setup, n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'head', animal_setup, ['nose', 'leftear', 'rightear', 'neck'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'hips', animal_setup, ['lefthip', 'tail', 'righthip'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'body', animal_setup, ['neck', 'lefthip', 'righthip', 'tail'], n_shifts = n_shifts, mode = mode)

    #####################
    #Appearance features#
    #####################

    ## absolute orientation of mice
    features_df = _compute_abs_angle(features_df, 'head_hips', animal_setup, ['centroid_head', 'centroid_hips'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'head_nose', animal_setup, ['neck', 'nose'], centroid = False, n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'tail_neck', animal_setup, ['tail', 'neck'], centroid = False, n_shifts = n_shifts, mode = mode)
    ## relative orientation of mice
    features_df = _compute_rel_angle(features_df, 'leftear_neck_rightear', animal_setup, ['leftear', 'neck', 'rightear'], n_shifts = n_shifts, mode = mode)
    ## major axis len, minor axis len of ellipse fit to mouses body
    features_df = _compute_ellipsoid(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    #####################
    #Locomotion features#
    #####################

    features_df = _compute_kinematics(features_df, ['all', 'head', 'hips', 'body'], animal_setup)

    #Intersection of union of bounding boxes of two mice
    features_df = _compute_iou(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    ## distance between all pairs of keypoints of each mouse
    features_df = make_features_distances(features_df, animal_setup)

    return features_df

def make_features_velocities(df, animal_setup, n_shifts = 5): # pragma: no cover

    bodypart_ids = animal_setup['bodypart_ids']
    mouse_ids = animal_setup['mouse_ids']
    colnames = animal_setup['colnames']

    features_df = df.copy()

    ##Make the distance features
    for i, bp1 in enumerate(bodypart_ids):
        for j, bp2 in enumerate(bodypart_ids):
            if i < j:
                for mouse_id in mouse_ids:
                    #We can compute the intra-mouse difference
                    f1x = '_'.join([mouse_id, 'x', bp1])
                    f2x = '_'.join([mouse_id, 'x', bp2])
                    f1y = '_'.join([mouse_id, 'y', bp1])
                    f2y = '_'.join([mouse_id, 'y', bp2])
                    f_new = '_'.join([mouse_id, 'speed', bp1, bp2])
                    features_df[f_new] = \
                        np.sqrt((features_df[f1x].diff(periods = n_shifts) - features_df[f2x].diff(periods = n_shifts))**2 + 
                                (features_df[f1y].diff(periods = n_shifts) - features_df[f2y].diff(periods = n_shifts))**2)
            #Inter-mouse difference
            f1x = '_'.join([mouse_ids[0], 'x', bp1])
            f2x = '_'.join([mouse_ids[1], 'x', bp2])
            f1y = '_'.join([mouse_ids[0], 'y', bp1])
            f2y = '_'.join([mouse_ids[1], 'y', bp2])
            f_new = '_'.join(['M0_M1', 'speed', bp1, bp2])
            features_df[f_new] = \
                        np.sqrt((features_df[f1x].diff(periods = n_shifts) - features_df[f2x].diff(periods = n_shifts))**2 + 
                                (features_df[f1y].diff(periods = n_shifts) - features_df[f2y].diff(periods = n_shifts))**2)

    #Remove base features
    features_df = features_df.drop(columns = colnames)

    ##Clean up seq_id columns
    features_df, _ = boiler_plate(features_df)

    return features_df

def make_features_social(df, animal_setup, n_shifts = 3, mode = 'shift'):

    features_df = df.copy()
    colnames = animal_setup['colnames']

    #######################
    ## Position features ##
    #######################
    features_df = _compute_centroid(features_df, 'all', animal_setup, n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'head', animal_setup, ['nose', 'leftear', 'rightear', 'neck'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'hips', animal_setup, ['lefthip', 'tail', 'righthip'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_centroid(features_df, 'body', animal_setup, ['neck', 'lefthip', 'righthip', 'tail'], n_shifts = n_shifts, mode = mode)

    #####################
    #Appearance features#
    #####################

    ## absolute orientation of mice
    features_df = _compute_abs_angle(features_df, 'head_hips', animal_setup, ['centroid_head', 'centroid_hips'], n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'head_nose', animal_setup, ['neck', 'nose'], centroid = False, n_shifts = n_shifts, mode = mode)
    features_df = _compute_abs_angle(features_df, 'tail_neck', animal_setup, ['tail', 'neck'], centroid = False, n_shifts = n_shifts, mode = mode)
    ## relative orientation of mice
    features_df = _compute_rel_angle(features_df, 'leftear_neck_rightear', animal_setup, ['leftear', 'neck', 'rightear'], n_shifts = n_shifts, mode = mode)

    ## major axis len, minor axis len of ellipse fit to mouses body
    features_df = _compute_ellipsoid(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    #################
    #Social features#
    #################

    #Added columns 
    added_cols = list(set(features_df.columns).difference(set(df.columns)))

    features_df = _compute_relative_body_motions(features_df, animal_setup, n_shifts = n_shifts, mode = mode)
    features_df = _compute_relative_body_angles(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    #Intersection of union of bounding boxes of two mice
    features_df = _compute_iou(features_df, animal_setup, n_shifts = n_shifts, mode = mode)

    features_df = features_df.drop(columns = added_cols)
    colnames = [c for c in colnames if c in features_df.columns]
    features_df = features_df.drop(columns = colnames)

    features_df, _ = boiler_plate(features_df)

    return features_df

def compute_mars_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    animal_setup = df.pose.animal_setup
    features_df = make_features_mars_distr(df[raw_col_names], animal_setup) 
    return features_df

def compute_distance_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    animal_setup = df.pose.animal_setup
    features_df = make_features_distances(df[raw_col_names], animal_setup) 
    return features_df

def compute_mars_reduced_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    animal_setup = df.pose.animal_setup
    features_df = make_features_mars_reduced(df[raw_col_names], animal_setup) 
    return features_df

def compute_social_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    animal_setup = df.pose.animal_setup
    features_df = make_features_social(df[raw_col_names], animal_setup) 
    return features_df

def compute_velocity_features(df : pd.DataFrame, raw_col_names : list, **kwargs) -> pd.DataFrame:
    animal_setup = df.pose.animal_setup
    features_df = make_features_velocities(df[raw_col_names], animal_setup) 
    return features_df
