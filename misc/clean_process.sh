kill $(ps aux | grep main_multi.py | grep -v grep | awk '{print $2}')
