<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `io`
Loading and saving tracking and behavior annotation files  

**Global Variables**
---------------
- **DEFAULT_BUFFER_SIZE**
- **SEEK_SET**
- **SEEK_CUR**
- **SEEK_END**
- **XY_IDS**

---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_DLC_tracks`

```python
read_DLC_tracks(
    fn_in:str,
    part_renamer:dict=None,
    animal_renamer:dict=None
) → tuple
```

Read in tracks from DLC. 



**Args:**
 
 - <b>`fn_in`</b>:  csv file that has DLC tracks 
 - <b>`part_renamer`</b>:  dictionary to rename body parts, if needed  
 - <b>`animal_renamer`</b>:  dictionary to rename animals, if needed 



**Returns:**
 Pandas DataFrame with (n_animals*2*n_body_parts) columns plus with filename and frame,   List of body parts,  List of animals,  Columns names for DLC tracks 


---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rename_df_cols`

```python
rename_df_cols(df:DataFrame, renamer:dict) → DataFrame
```

Rename dataframe columns  



**Args:**
 
 - <b>`df`</b>:  Pandas dataframe whose columns to rename 
 - <b>`renamer`</b>:  dictionary whose key:value pairs define the substitutions to make 



**Returns:**
 The dataframe with renamed columns. 


---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_DLC_tracks_h5`

```python
save_DLC_tracks_h5(df:DataFrame, fn_out:str) → None
```

Save DLC tracks in h5 format. 



**Args:**
 
 - <b>`df`</b>:  Pandas dataframe to save 
 - <b>`fn_out`</b>:  Where to save the dataframe 


---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_data`

```python
load_data(fn:str)
```

Load an object from a pickle file 



**Args:**
 
 - <b>`fn`</b>:  The filename 



**Returns:**
 The pickled object. 


---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_boris_annotation`

```python
read_boris_annotation(fn_in:str, fps:int, duration:float) → ndarray
```

Read behavior annotation from BORIS exported csv file 



**Args:**
 
 - <b>`fn_in`</b>:  The filename with BORIS behavior annotations to load 
 - <b>`fps`</b>:  Frames per second of video 
 - <b>`duration`</b>:  Duration of video 



**Returns:**
 A numpy array which indicates, for all frames, if behavior is occuring (1) or not (0) 


---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BufferedIOBase`
Base class for buffered IO objects. 

The main difference with RawIOBase is that the read() method supports omitting the size argument, and does not have a default implementation that defers to readinto(). 

In addition, read(), readinto() and write() may raise BlockingIOError if the underlying raw stream is in non-blocking mode and not ready; unlike their raw counterparts, they will never return None. 

A typical implementation should not inherit from a RawIOBase implementation, but wrap one. 





---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RawIOBase`
Base class for raw binary I/O. 





---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TextIOBase`
Base class for text I/O. 

This class provides a character and line based interface to stream I/O. There is no readinto method because Python's character strings are immutable. There is no public constructor. 





---

<a href="https://github.com/benlansdell/behaveml/tree/master/behaveml/io.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `UnsupportedOperation`










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
