kill $(ps aux | grep main_r.py | grep -v grep | awk '{print $2}')
