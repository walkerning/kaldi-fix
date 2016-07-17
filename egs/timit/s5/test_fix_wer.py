#!/usr/bin/env python2.7

import subprocess
import tempfile
import sys

phases = sys.argv[1:]
if not phases:
    print "usage: python test_fix_wer.py [phases...]"
    sys.exit(1)

TEST_CASES = [
    None,
    # ("<DynamicFixedPoint>", # strategy
    #  4, # default_blob_bit
    #  4, # default_param_bit
    #  8, # first_blob_bit
    #  8 ), # first_param_bit
    # ("<DynamicFixedPoint>",
    #  4, # default_blob_bit
    #  4, # default_param_bit
    #  16, # first_blob_bit
    #  16 ), # first_param_bit
    ("<DynamicFixedPoint>",
     8, # default_blob_bit
     8, # default_param_bit
     8, # first_blob_bit
     8 ), # first_param_bit
    ("<DynamicFixedPoint>",
     8, # default_blob_bit
     8, # default_param_bit
     12, # first_blob_bit
     12 ), # first_param_bit
    ("<DynamicFixedPoint>",
     8, # default_blob_bit
     8, # default_param_bit
     16, # first_blob_bit
     16 ), # first_param_bit
    ("<DynamicFixedPoint>",
     10, # default_blob_bit
     10, # default_param_bit
     10, # first_blob_bit
     10 ), # first_param_bit
    ("<DynamicFixedPoint>",
     10, # default_blob_bit
     10, # default_param_bit
     12, # first_blob_bit
     12 ), # first_param_bit
    ("<DynamicFixedPoint>",
     12, # default_blob_bit
     12, # default_param_bit
     12, # first_blob_bit
     12 ), # first_param_bit
    # ("<DynamicFixedPoint>",
    #  12, # default_blob_bit
    #  12, # default_param_bit
    #  14, # first_blob_bit
    #  14 ), # first_param_bit
    ("<DynamicFixedPoint>",
     14, # default_blob_bit
     14, # default_param_bit
     14, # first_blob_bit
     14 ), # first_param_bit
    ("<DynamicFixedPoint>",
     16, # default_blob_bit
     16, # default_param_bit
     16, # first_blob_bit
     16 ), # first_param_bit
]

_, fname = tempfile.mkstemp(suffix="kaldi-fix-test")

for case in TEST_CASES:
    config_line = "{} <DefaultBlobBit> {} <DefaultParamBit> {} <BlobIndexBit> 0 {} <ParamIndexBit> 0 {}\n".format(*case) if case is not None else ""
    print ",\t".join(str(x) for x in case) if case is not None else "Float point",
    with open(fname, "w") as f:
        f.write(config_line)
    for phase in phases:
        try:
            wer = subprocess.check_output("DEBUG=0 bash ./test_wer.sh --phase {} --fixconf {} | cut -d' ' -f2".format(phase, fname), shell=True)
            print "\t" + wer.strip()
        except subprocess.CalledProcessError as e:
            print e.output
