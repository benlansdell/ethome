import numpy as np 
import keras 
from collections import Counter 
import pandas as pd 

#from lib.helper import colnames 
from .feature_engineering import make_features_mars, make_features_mars_distr

def make_df(pts, colnames = None):
    df = []
    for idx in range(len(pts)):
        data = pts[idx].flatten()
        df.append(list(data))
    if colnames:
        return pd.DataFrame(df, columns = colnames)
    else:
        return pd.DataFrame(df)

def features_identity(inputs):
    return inputs, inputs.shape[1:]

def features_via_sklearn(inputs, featurizer):
    #Use the ML functions to turn this into a pandas data table
    df = make_df(inputs)
    features_df, _, _ = featurizer(df)
    features = np.array(features_df)
    return features, features.shape

features_mars = lambda x: features_via_sklearn(x, make_features_mars)

#features_mars_no_shift = lambda x: features_via_sklearn(x, make_features_mars_no_shift)

features_mars_distr = lambda x: features_via_sklearn(x, make_features_mars_distr)

def features_distances(inputs):

    #inputs.shape (4509, 2,7,2) = (frame, mouse ID, body part, x/y)

    features = []    
    ##Make the distance features
    for i in range(7):
        for j in range(7):
            if i < j:
                for mouse_id in range(2):
                    #We can compute the intra-mouse difference
                    f1x = inputs[:,mouse_id,i,0]
                    f2x = inputs[:,mouse_id,j,0]
                    f1y = inputs[:,mouse_id,i,1]
                    f2y = inputs[:,mouse_id,j,1]
                    features.append(np.sqrt((f1x - f2x)**2 + (f1y - f2y)**2))
            #Inter-mouse difference
            f1x = inputs[:,0,i,0]
            f2x = inputs[:,1,j,0]
            f1y = inputs[:,0,i,1]
            f2y = inputs[:,1,j,1]
            features.append(np.sqrt((f1x - f2x)**2 + (f1y - f2y)**2))

    features = np.vstack(features).T

    return features, features.shape[1:]

def features_distances_normalized(inputs):

    #inputs.shape (4509, 2,7,2) = (frame, mouse ID, body part, x/y)

    features = []    
    for i in range(7):
        for j in range(7):
            if i < j:
                for mouse_id in range(2):
                    #Compute the intra-mouse difference
                    f1x = inputs[:,mouse_id,i,0]
                    f2x = inputs[:,mouse_id,j,0]
                    f1y = inputs[:,mouse_id,i,1]
                    f2y = inputs[:,mouse_id,j,1]
                    distances = np.sqrt((f1x - f2x)**2 + (f1y - f2y)**2)
                    distances /= np.linalg.norm(distances)
                    features.append(distances)
            #Inter-mouse difference
            f1x = inputs[:,0,i,0]
            f2x = inputs[:,1,j,0]
            f1y = inputs[:,0,i,1]
            f2y = inputs[:,1,j,1]
            distances = np.sqrt((f1x - f2x)**2 + (f1y - f2y)**2)
            features.append(distances)

    features = np.vstack(features).T

    return features, features.shape[1:]
    #output (4509, X) = (frame, feature) 

class MABe_Generator(keras.utils.Sequence):
    def __init__(self, pose_dict, 
                 batch_size, dim, 
                 use_conv, num_classes, augment=False,
                 class_to_number=None,
                 past_frames=0, future_frames=0, 
                 frame_gap=1, shuffle=False,
                 mode='fit', featurize = features_identity):
        self.batch_size = batch_size
        self.featurize = featurize
        self.video_keys = list(pose_dict.keys())
        self.dim = dim
        self.use_conv = use_conv
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.frame_gap = frame_gap
        self.shuffle = shuffle
        self.num_classes=num_classes
        self.augment = augment
        self.mode = mode

        self.class_to_number = class_to_number

        self.video_indexes = []
        self.frame_indexes = []
        self.X = {}
        if self.mode == 'fit':
            self.y = []
        self.pad = self.past_frames * self.frame_gap
        future_pad = self.future_frames * self.frame_gap
        pad_width = (self.pad, future_pad), (0, 0), (0, 0), (0, 0)
        self.seq_lengths = {}
        for vc, key in enumerate(self.video_keys):
            if self.mode == 'fit':
                anno = pose_dict[key]['annotations']
                self.y.extend(anno)
            nframes = len(pose_dict[key]['keypoints'])
            self.video_indexes.extend([vc for _ in range(nframes)])
            self.frame_indexes.extend(range(nframes))
            self.X[key],_ = self.featurize(np.pad(pose_dict[key]['keypoints'], pad_width))
            self.seq_lengths[key] = nframes
        
        if self.mode == 'fit':
            self.y = np.array(self.y)
            #Compute class weights:
            class_weights = np.zeros(self.num_classes)
            counts = Counter(self.y)
            for c in range(num_classes):
                class_weights[c] = counts[c]
            class_weights /= np.sum(class_weights)
            self.class_weights = class_weights
        else:
            self.class_weights = np.ones(num_classes)
        
        self.X_dtype = self.X[key].dtype

        self.indexes = list(range(len(self.frame_indexes)))

        if self.mode == 'predict':
            extra_predicts = -len(self.indexes) % self.batch_size # So that last part is not missed
            self.indexes.extend(self.indexes[:extra_predicts])
            self.indexes = np.array(self.indexes)
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def augment_fn(self, x):
        # Rotate
        angle = (np.random.rand()-0.5) * (np.pi * 2)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        x = np.dot(x, rot)

        # Shift - All get shifted together
        shift = (np.random.rand(2)-0.5) * 2 * 0.25
        x = x + shift
        return x

    def __getitem__(self, index):
        bs = self.batch_size
        indexes = self.indexes[index*bs:(index+1)*bs]
        X = np.empty((bs, *self.dim), self.X_dtype)
        if self.mode == 'predict':
            vkey_fi_list = []
        for bi, idx in enumerate(indexes):
            vkey = self.video_keys[self.video_indexes[idx]]
            fi = self.frame_indexes[idx]
            if self.mode == 'predict':
                vkey_fi_list.append((vkey, fi))
            fi = fi + self.pad
            start = fi - self.past_frames*self.frame_gap
            stop = fi + (self.future_frames + 1)*self.frame_gap
            assert start >= 0

            Xi = self.X[vkey][start:stop:self.frame_gap].copy()
          
            if self.augment:
                Xi = self.augment_fn(Xi)
            X[bi] = np.reshape(Xi, self.dim)

        if self.mode == 'fit':
            y_vals = self.y[indexes]
            # Converting to one hot because F1 callback needs one hot
            y = np.zeros( (bs,self.num_classes), np.float32)
            y[np.arange(bs), y_vals] = 1
            return X, y

        elif self.mode == 'predict':
            return X, vkey_fi_list

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)