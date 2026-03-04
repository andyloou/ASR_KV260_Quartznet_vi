#!/bin/bash
fuser -k 5000/tcp 2>/dev/null
sleep 1

export GLOG_minloglevel=3
export XLNX_ENABLE_SKIP_FATAL=1

export PYTHONWARNINGS="ignore"

python app2.py 2> >(grep -v "pass_main\|skip this subgraph\|XLNX_ENABLE_SKIP_FATAL\|Catch fatal" >&2)
