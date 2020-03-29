#!/usr/bin/env python3

import sys
try:
    if sys.argv[1] == 'wq':
        print('wine!')
    elif sys.argv[1] == 'gb':
        print('blobs!')
    else:
        raise Exception
except Exception:
    print("please provide 'wq' or 'gb' as an argument")
    exit(1)
