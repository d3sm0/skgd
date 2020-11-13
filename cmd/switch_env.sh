#!/usr/bin/env bash

VENDOR=$(grep vendor_id /proc/cpuinfo -m 1 | awk '{print $3}')
echo "found vendor $VENDOR"
if [[ $VENDOR == *"Intel"* ]]; then
	echo "source intel env"
	SOURCE_DIR=~/microgrid_env_intel/bin/activate
else
	echo "source amd env"
	SOURCE_DIR=~/microgrid_env_amd/bin/activate
fi
