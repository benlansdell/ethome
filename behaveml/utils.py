def _exec_php(cmd):
    from subprocess import Popen, PIPE, STDOUT
    p = Popen(cmd, shell=False, stdout=PIPE, stderr=STDOUT)
    return [p.wait(), p.stdout.readlines()]
    
def checkFFMPEG() -> bool:
    """
    Check for ffmpeg dependencies

    Returns:
        True if can find ffmpeg in path, false otherwise
    """
    try:
        return_value = _exec_php(['ffmpeg', '-version', '2>&1'])[0]
        if return_value == 0: return True
    except Exception:
        pass

    return False 