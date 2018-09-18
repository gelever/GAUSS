# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of smoothG. For more information and source code
# availability, see https://www.github.com/llnl/smoothG.
#
# smoothG is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #

"""
A way to interface some basic JSON-parsing tests in python
with the cmake/ctest testing framework.

Stephan Gelever
gelever1@llnl.gov
17 July 2017
"""

from __future__ import print_function

import subprocess

import sys
import platform
import json

import random
import math

spe10_perm_file = "@SPE10_PERM@"

memorycheck_command = "@MEMORYCHECK_COMMAND@"

test_tol = float("@GAUSS_TEST_TOL@")
num_procs = "@GAUSS_TEST_PROCS@"


def run_test(command, expected={}, verbose=False):
    """ Executes test

    Args:
        command:    command to run test
        expected:   expected result of test
        verbose:    display additional info

    Returns:
        bool:       true if test passes

    """
    if verbose:
        print(command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = p.communicate()

    if verbose:
        print(stdout)
        print(stderr)

    if p.returncode != 0:
        return False

    output = json_parse_lines(stdout.splitlines())

    for key, expected_val in expected.items():
        test_val = output[key]

        if abs(float(expected_val) - float(test_val)) > test_tol:
            return False

    return True


def json_parse_lines(lines, max_depth=10, max_height=6):
    """ Look for a JSON object on the last few lines of input

    Args:
        lines:  lines to parse
        max_depth:   maximum number of lines to parse from end
        max_height:   maximum number of lines to check as JSON object

    Returns:
        dict: parsed json object

    """
    for index in range(-1, -max_depth, -1):
        for i in range(max_height):
            try:
                name = "".join(lines[index - i:])
                return json.loads(name)
            except ValueError:
                pass
    return {}


def make_tests():
    """ Generates test dictionary

    Tests are the following format:
        - dictionary key is test name
        - dictionary value is an array containing:
            - command to execute
            - expected result [optional]

    Returns:
        dict:     collection of tests

    """
    tests = dict()

    tests["eigenvector1"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "1"],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0372402200489}]

    tests["eigenvector4"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "4"],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.582594743276283}]

    tests["fv-hybridization"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "4", "-hb"],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.36479217603912}]

    tests["slice19"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "1", "--slice", "19"],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0372402200489}]

    tests["fv-metis"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "1", "-ma"],
         {"finest-div-error": 0.5640399149852695,
          "finest-p-error": 0.1738574977361006,
          "finest-u-error": 0.2978586987874174,
          "operator-complexity": 1.072555012224939}]

    tests["ml_eigenvector1"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "1"],
         {"finest-div-error": 0.8660254036696198,
          "finest-p-error": 0.5641246228439858,
          "finest-u-error": 0.7197949543870888,
          "operator-complexity": 1.049633251833741}]

    tests["ml_eigenvector4"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "4"],
         {"finest-div-error": 0.7075035723146266,
          "finest-p-error": 0.1278078697176256,
          "finest-u-error": 0.2643569166150334,
          "operator-complexity": 1.775748777506112}]

    tests["ml_fv-hybridization"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "4", "-hb"],
         {"finest-div-error": 0.7075035723146259,
          "finest-p-error": 0.1278078697147398,
          "finest-u-error": 0.2643569166439237,
          "operator-complexity": 1.496332518337408}]

    tests["ml_slice19"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "1"],
         {"finest-div-error": 0.8660254037170271,
          "finest-p-error": 0.4970818098461737,
          "finest-u-error": 0.5957025563910179,
          "operator-complexity": 1.049633251833741}]

    tests["ml_fv-metis"] = \
        [["./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "1", "-ma", "-np", "132"],
         {"finest-div-error": 0.8677379997088861,
          "finest-p-error": 0.2834920562283691,
          "finest-u-error": 0.4925040461853005,
          "operator-complexity": 1.087438875305623}]

    tests["timestep"] = \
        [["./timestep",
          "--perm", spe10_perm_file,
          "-time", "100.0"]]

    tests["pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "1"],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0372402200488997}]

    tests["pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "4"],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.5825947432762835}]

    tests["parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "4", "-hb"],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.3647921760391197}]

    tests["parslice19"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-t", "1.0", "-m", "1", "--slice", "19"],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0372402200488997}]

    tests["ml_pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "1"],
         {"finest-div-error": 0.8660254036840376,
          "finest-p-error": 0.226030077744117,
          "finest-u-error": 0.4167023405052202,
          "operator-complexity": 1.050412591687041}]

    tests["ml_pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "4"],
         {"finest-div-error": 0.7657991607437392,
          "finest-p-error": 0.07625579392353324,
          "finest-u-error": 0.191648625052069,
          "operator-complexity": 1.787897310513447}]

    tests["ml_parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "4", "-hb"],
         {"finest-div-error": 0.7657991607437392,
          "finest-p-error": 0.07625579392353324,
          "finest-u-error": 0.191648625052069,
          "operator-complexity": 1.505867970660147}]

    tests["ml_parslice19"] = \
        [["mpirun", "-n", num_procs, "./finitevolume",
          "--perm", spe10_perm_file,
          "-nl", "3", "-t", "1.0", "-m", "1"],
         {"finest-div-error": 0.8660254037589696,
          "finest-p-error": 0.5257859517082242,
          "finest-u-error": 0.6835573888435186,
          "operator-complexity": 1.050412591687041}]

    tests["partimestep"] = \
        [["mpirun", "-n", num_procs, "./timestep",
          "--perm", spe10_perm_file,
          "-time", "100.0"]]

    if "tux" in platform.node():
        tests["veigenvector"] = \
            [["mpirun", "-n", num_procs,
              memorycheck_command, "--leak-check=full",
              "./finitevolume",
              "--perm", spe10_perm_file,
              "-t", "1.0", "-m", "1"]]

    return tests


def run_all_tests(tests, verbose=False):
    """ Execute all tests and display results

    Any exception raised during a test counts as a
    failure

    Args:
        tests (dict):    tests to perform,
                         see make_tests for format

    Returns:
        int:     number of failed tests

    """
    totaltests = len(tests)
    success = 0

    for i, (name, test) in enumerate(tests.items()):
        try:
            result = run_test(*test, verbose=verbose)
        except BaseException as err:
            print("{0} Failed: {1}".format(name, err))
            result = False

        success += result
        status = "passed" if result else "FAILED"

        print("  ({0}/{1}) [{2}] {3}.".format(i + 1, totaltests, name, status))

    failures = totaltests - success

    print("Ran {0} tests with {1} successes and {2} failures.".format(
        totaltests, success, failures))

    return failures


def main(argv):
    """ Parses command line options and runs tests

    Empty commandline runs all tests
    Otherwise individual tests can be specified by name
    Pass in '-nv' with args to remove additional information

    Args:
        argv (list):     command line parameters

    Returns:
        int:     number of failed tests

    """
    verbose = True

    if "-nv" in argv:
        verbose = False
        argv.remove("-nv")

    if "-np" in argv:
        global num_procs
        num_procs = argv[argv.index("-np") + 1]
        argv.remove("-np")
        argv.remove(num_procs)

    tests = make_tests()

    if argv:
        tests = dict((name, tests[name]) for name in argv if name in tests)

    return run_all_tests(tests, verbose)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
