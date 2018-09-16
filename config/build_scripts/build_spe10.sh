#!/bin/sh

# Downloads and builds SPE10 dataset
if [ -z "$INSTALL_DIR" ]; then INSTALL_DIR=${PWD}/extern; fi

TMP_DIR=/tmp/spe10

mkdir -p $TMP_DIR
cd $TMP_DIR

wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" \
    http://www.spe.org/web/csp/datasets/por_perm_case2a.zip
unzip -d ${INSTALL_DIR}/spe10 por_perm_case2a.zip

rm -rf $TMP_DIR
