kill $(ps aux | grep main_single_2.py | grep -v grep | awk '{print $2}')
