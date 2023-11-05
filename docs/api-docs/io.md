<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `io`
Loading and saving tracking and behavior annotation files  

**Global Variables**
---------------
- **DEFAULT_BUFFER_SIZE**
- **SEEK_SET**
- **SEEK_CUR**
- **SEEK_END**
- **XY_IDS**
- **XYLIKELIHOOD_IDS**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `uniquifier`

```python
uniquifier(seq)
```

Return a sequence (e.g. list) with unique elements only, but maintaining original list order 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_sklearn_model`

```python
save_sklearn_model(model, fn_out)
```

Save sklearn model to file 



**Args:**
 
 - <b>`model`</b>:  sklearn model to save 
 - <b>`fn_out`</b>:  filename to save to 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_sklearn_model`

```python
load_sklearn_model(fn_in)
```

Load sklearn model from file 



**Args:**
 
 - <b>`fn_in`</b>:  filename to load from 



**Returns:**
 the loaded sklearn model 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_DLC_tracks`

```python
read_DLC_tracks(
    fn_in: str,
    part_renamer: dict = None,
    animal_renamer: dict = None,
    read_likelihoods: bool = True,
    labels: DataFrame = None
) → tuple
```

Read in tracks from DLC. 



**Args:**
 
 - <b>`fn_in`</b>:  csv file that has DLC tracks 
 - <b>`part_renamer`</b>:  dictionary to rename body parts, if needed  
 - <b>`animal_renamer`</b>:  dictionary to rename animals, if needed 
 - <b>`read_likelihoods`</b>:  default True. Whether to attach DLC likelihoods to table 



**Returns:**
 Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame,   List of body parts,  List of animals,  Columns names for DLC tracks (excluding likelihoods, if read in),  Scorer 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_NWB_tracks`

```python
read_NWB_tracks(
    fn_in: str,
    part_renamer: dict = None,
    animal_renamer: dict = None,
    read_likelihoods: bool = True
) → tuple
```

Read in tracks from NWB PoseEstimiationSeries format (something saved using the DLC2NWB package). 



**Args:**
 
 - <b>`fn_in`</b>:  nwb file that has the tracking information 
 - <b>`part_renamer`</b>:  dictionary to rename body parts, if needed  
 - <b>`animal_renamer`</b>:  dictionary to rename animals, if needed 
 - <b>`read_likelihoods`</b>:  default True. Whether to attach DLC likelihoods to table 



**Returns:**
 Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame,   List of body parts,  List of animals,  Columns names for pose tracks (excluding likelihoods, if read in),  Scorer 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_DLC_tracks_h5`

```python
save_DLC_tracks_h5(df: DataFrame, fn_out: str) → None
```

Save DLC tracks in h5 format. 



**Args:**
 
 - <b>`df`</b>:  Pandas dataframe to save 
 - <b>`fn_out`</b>:  Where to save the dataframe 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_data`

```python
load_data(fn: str)
```

Load an object from a pickle file 



**Args:**
 
 - <b>`fn`</b>:  The filename 



**Returns:**
 The pickled object. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sample_nwb_paths`

```python
get_sample_nwb_paths()
```

Get path to a sample NWB file with tracking data for testing and dev purposes. 



**Returns:**
  Path to a sample NWB file. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sample_data_paths_dlcboris`

```python
get_sample_data_paths_dlcboris()
```

Get path to sample data files provided with package.  



**Returns:**
  (tuple) list of DLC tracking file, list of boris annotation files 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sample_data`

```python
get_sample_data()
```

Load a sample dataset of 5 mice social interaction videos. Each video is approx. 5 minutes in duration 



**Returns:**
  (ExperimentDataFrame) Data frame with the corresponding tracking and behavior annotation files 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_boris_annotation`

```python
read_boris_annotation(
    fn_in: str,
    fps: int,
    duration: float,
    behav_labels: dict = None
) → tuple
```

Read behavior annotation from BORIS exported csv file.  

This will import behavior types specified (or all types, if behavior_list is None) and assign a numerical label to each. Overlapping annotations (those occurring simulataneously) are not supported. Any time the video is annotated as being in multiple states, the last state will be the one labeled. 



**Args:**
 
 - <b>`fn_in`</b>:  The filename with BORIS behavior annotations to load 
 - <b>`fps`</b>:  The frames per second of the video 
 - <b>`duration`</b>:  The duration of the video in seconds 
 - <b>`behav_labels`</b>:  If provided, only import behaviors with these names. Default = None = import everything.  



**Returns:**
 A numpy array which indicates, for all frames, which behavior is occuring. 0 = no behavior, 1 and above are the labels of the behaviors. A dictionary with keys the numerical labels and values the names of the behaviors.  


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py#L394"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_behavior_labels`

```python
create_behavior_labels(boris_files)
```

Create behavior labels from BORIS exported csv files. 



**Args:**
 
 - <b>`boris_files`</b>:  List of BORIS exported csv files 



**Returns:**
 A dictionary with keys the numerical labels and values the names of the behaviors. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BufferedIOBase`
Base class for buffered IO objects. 

The main difference with RawIOBase is that the read() method supports omitting the size argument, and does not have a default implementation that defers to readinto(). 

In addition, read(), readinto() and write() may raise BlockingIOError if the underlying raw stream is in non-blocking mode and not ready; unlike their raw counterparts, they will never return None. 

A typical implementation should not inherit from a RawIOBase implementation, but wrap one. 





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `IOBase`
The abstract base class for all I/O classes, acting on streams of bytes. There is no public constructor. 

This class provides dummy implementations for many methods that derived classes can override selectively; the default implementations represent a file that cannot be read, written or seeked. 

Even though IOBase does not declare read, readinto, or write because their signatures will vary, implementations and clients should consider those methods part of the interface. Also, implementations may raise UnsupportedOperation when operations they do not support are called. 

The basic type used for binary data read from or written to a file is bytes. Other bytes-like objects are accepted as method arguments too. In some cases (such as readinto), a writable object is required. Text I/O classes work with str data. 

Note that calling any method (except additional calls to close(), which are ignored) on a closed stream should raise a ValueError. 

IOBase (and its subclasses) support the iterator protocol, meaning that an IOBase object can be iterated over yielding the lines in a stream. 

IOBase also supports the :keyword:`with` statement. In this example, fp is closed after the suite of the with statement is complete: 

with open('spam.txt', 'r') as fp:  fp.write('Spam and eggs!') 





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RawIOBase`
Base class for raw binary I/O. 





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TextIOBase`
Base class for text I/O. 

This class provides a character and line based interface to stream I/O. There is no readinto method because Python's character strings are immutable. There is no public constructor. 





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnsupportedOperation`










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
