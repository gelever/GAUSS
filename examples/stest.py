# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-759464. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of GAUSS. For more information and source code
# availability, see https://www.github.com/gelever/GAUSS.
#
# GAUSS is free software; you can redistribute it and/or modify it under the
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
graph_data = "@PROJECT_SOURCE_DIR@/graphdata"

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
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0372402200489}]

    tests["eigenvector4"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.582594743276283}]

    tests["fv-hybridization"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.36479217603912}]

    tests["slice19"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_19.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0372402200489}]

    tests["fv-metis"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "1", "--ma", "--np", "132"],
         {"finest-div-error": 0.5888006112802964,
          "finest-p-error": 0.1947684753679836,
          "finest-u-error": 0.3215799309244913,
          "operator-complexity": 1.072906479217604}]

    tests["ml_eigenvector1"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.8660254036696198,
          "finest-p-error": 0.5641246228439858,
          "finest-u-error": 0.7197949543870888,
          "operator-complexity": 1.049633251833741}]

    tests["ml_eigenvector4"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 0.7075035723146266,
          "finest-p-error": 0.1278078697176256,
          "finest-u-error": 0.2643569166150334,
          "operator-complexity": 1.775748777506112}]

    tests["ml_fv-hybridization"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 0.7075035723146259,
          "finest-p-error": 0.1278078697147398,
          "finest-u-error": 0.2643569166439237,
          "operator-complexity": 1.496332518337408}]

    tests["ml_slice19"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_19.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.8660254037170271,
          "finest-p-error": 0.4970818098461737,
          "finest-u-error": 0.5957025563910179,
          "operator-complexity": 1.049633251833741}]

    tests["ml_fv-metis"] = \
        [["./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "1", "--ma", "--np", "132"],
         {"finest-div-error": 0.8677379997088861,
          "finest-p-error": 0.2834920562283691,
          "finest-u-error": 0.4925040461853005,
          "operator-complexity": 1.087438875305623}]

    tests["samplegraph1"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.37918423747873353,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.040268292682927}]

    tests["graph-metis"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "1", "--ma"],
         {"finest-div-error": 0.44710819907744104,
          "finest-p-error": 0.44939226988126274,
          "finest-u-error": 0.42773807524771068,
          "operator-complexity": 1.040268292682927}]

    tests["graph-metis-mac"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "1", "--ma"],
         {"finest-div-error": 0.22228470008233389,
          "finest-p-error": 0.22265174467689006,
          "finest-u-error": 0.22168973853676807,
          "operator-complexity": 1.040268292682927}]

    tests["samplegraph4"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.6290000000000000}]

    tests["graph-hybridization"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 0.12051328492652449,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.540878048780488}]

    tests["graph-usegenerator"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "4", "--gg"],
         {"finest-div-error": 0.11283262603381641,
          "finest-p-error": 0.1203852548326301,
          "finest-u-error": 0.16674213482507089,
          "operator-complexity": 1.6290000000000000}]

    tests["graph-usegenerator-mac"] = \
        [["./generalgraph",
          "--t", "1.0", "--m", "4", "--gg"],
         {"finest-div-error": 0.1070681247529167,
          "finest-p-error": 0.10863131137013603,
          "finest-u-error": 0.12848813745253315,
          "operator-complexity": 1.2578874211257887}]

    tests["poweriter"] = \
        [["./poweriter"],
         {"coarse-error": 0.2060400964805032,
          "coarse-eval": 0.1766365064174485,
          "fine-error": 0.0029137744885041,
          "fine-eval": 0.1754552537017609}]

    tests["graph-weight"] = \
        [["./generalgraph",
          "--g", graph_data + "/vertex_edge_tiny.txt",
          "--w", graph_data + "/tiny_weights.txt",
          "--gf", "--ma", "--np", "2"],
         {"finest-div-error": 0.3033520464019937,
          "finest-p-error": 0.31217311873637132,
          "finest-u-error": 0.14767829457535478,
          "operator-complexity": 1.3000000000000000}]

    tests["timestep"] = \
        [["./graph_timestep",
          "--time", "100.0"]]

    tests["pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 2.0312444586906591e-08,
          "finest-p-error": 0.14743131732550618,
          "finest-u-error": 0.22621045683612057,
          "operator-complexity": 1.0372402200488997}]

    tests["pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 2.0336350399372878e-08,
          "finest-p-error": 0.05516198497834629,
          "finest-u-error": 0.052317636963252999,
          "operator-complexity": 1.5825947432762835}]

    tests["parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 1.3301680521537587e-08,
          "finest-p-error": 0.055161984984368362,
          "finest-u-error": 0.052317636981330032,
          "operator-complexity": 1.3647921760391197}]

    tests["parslice19"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_19.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 1.2837519341678676e-08,
          "finest-p-error": 0.23763409361749516,
          "finest-u-error": 0.16419932734829923,
          "operator-complexity": 1.0372402200488997}]

    tests["ml_pareigenvector1"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.8660254036840376,
          "finest-p-error": 0.226030077744117,
          "finest-u-error": 0.4167023405052202,
          "operator-complexity": 1.050412591687041}]

    tests["ml_pareigenvector4"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 0.7657991607437392,
          "finest-p-error": 0.07625579392353324,
          "finest-u-error": 0.191648625052069,
          "operator-complexity": 1.787897310513447}]

    tests["ml_parfv-hybridization"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_0.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 0.7657991607437392,
          "finest-p-error": 0.07625579392353324,
          "finest-u-error": 0.191648625052069,
          "operator-complexity": 1.505867970660147}]

    tests["ml_parslice19"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--g", graph_data + "/fe_vertex_edge.txt",
          "--w", graph_data + "/fe_weight_19.txt",
          "--p", graph_data + "/fe_part.txt",
          "--f", graph_data + "/fe_rhs.txt",
          "--nl", "3",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.8660254037589696,
          "finest-p-error": 0.5257859517082242,
          "finest-u-error": 0.6835573888435186,
          "operator-complexity": 1.050412591687041}]

    tests["parsamplegraph1"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "1"],
         {"finest-div-error": 0.37918423727222522,
          "finest-p-error": 0.38013398274257243,
          "finest-u-error": 0.38079825403520218,
          "operator-complexity": 1.0402682926829268}]

    tests["pargraph-metis"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "1", "--ma"],
         {"finest-div-error": 0.44710819906667049,
          "finest-p-error": 0.44939226988126274,
          "finest-u-error": 0.42773807524771068,
          "operator-complexity": 1.0402682926829268}]

    tests["pargraph-metis-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "1", "--ma"],
         {"finest-div-error": 0.22228470008233389,
          "finest-p-error": 0.22265174467689006,
          "finest-u-error": 0.22168973853676807,
          "operator-complexity": 1.040268292682927}]

    tests["parsamplegraph4"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "4"],
         {"finest-div-error": 0.12043046187567592,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.6290000000000000}]

    tests["pargraph-hybridization"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "4", "--hb"],
         {"finest-div-error": 0.12051328492652449,
          "finest-p-error": 0.13514675917148347,
          "finest-u-error": 0.19926779054787247,
          "operator-complexity": 1.5408780487804878}]

    tests["pargraph-usegenerator"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "4", "--gg"],
         {"finest-div-error": 0.11283262603381641,
          "finest-p-error": 0.1203852548326301,
          "finest-u-error": 0.16674213482507089,
          "operator-complexity": 1.6290000000000000}]

    tests["pargraph-usegenerator-mac"] = \
        [["mpirun", "-n", num_procs, "./generalgraph",
          "--t", "1.0", "--m", "4", "--gg"],
         {"finest-div-error": 0.1070681247529167,
          "finest-p-error": 0.10863131137013603,
          "finest-u-error": 0.12848813745253315,
          "operator-complexity": 1.2578874211257887}]

    tests["parpoweriter"] = \
        [["mpirun", "-n", num_procs, "./poweriter"],
         {"coarse-error": 0.2060400964805032,
          "coarse-eval": 0.1766365064174485,
          "fine-error": 0.0029137744885041,
          "fine-eval": 0.1754552537017609}]

    tests["partimestep"] = \
        [["mpirun", "-n", num_procs, "./graph_timestep",
          "--time", "100.0"]]

    # tests["isolate-coarsen"] = \
    #     [["./generalgraph",
    #       "--spect-tol", "1.0",
    #       "--max-evects", "4",
    #       "--metis-agglomeration",
    #       "--isolate", "0"],
    #      {"operator-complexity": 1.2736672633273667}]

    if "tux" in platform.node():
        tests["veigenvector"] = \
            [["mpirun", "-n", num_procs,
              memorycheck_command, "--leak-check=full",
              "./generalgraph",
              "--g", graph_data + "/fe_vertex_edge.txt",
              "--w", graph_data + "/fe_weight_0.txt",
              "--p", graph_data + "/fe_part.txt",
              "--f", graph_data + "/fe_rhs.txt",
              "--t", "1.0", "--m", "1"]]

        tests["vgraph-small-usegenerator"] = \
            [[memorycheck_command, "--leak-check=full",
              "./generalgraph",
              "--nv", "20", "--md", "4", "--t", "1.0",
              "--m", "1", "--gg"]]

        tests["vgraph-small-usegenerator-hb"] = \
            [[memorycheck_command, "--leak-check=full",
              "./generalgraph",
              "--nv", "20", "--md", "4", "--t", "1.0",
              "--m", "1", "--gg", "--hb"]]

    return tests


def stress_test(num_tests=1, verbose=True):
    """ Generate random problem and solve

    Args:
        num_test (int):    number of tests to perform

    Returns:
        int:     number of failed tests
    """
    failed_tests = []

    for i in range(int(num_tests)):
        nv = random.randrange(50, 10000)
        b = random.uniform(0.00, 0.20)
        m = random.randrange(1, 5)
        t = random.uniform(0.00, 0.005)
        np = random.randrange(5, max(10, int(nv / random.randrange(10, 50))))
        md = int(math.log(nv) + random.randrange(1, 5))

        if md % 2 > 0:
            md += 1

        for proc in range(min(np, int(num_procs))):
            print("Test:\t{0}.{1}".format(i, proc + 1), end='\r')

            test = [["mpirun", "-np", str(proc + 1), "./generalgraph",
                "--nv", str(nv), "--t", str(t), "--m", str(m), "--md", str(md),
                "--gg", "--gf", "--ma", "--np", str(np), "--b", str(b), "--s", "-1"]]

            if not run_test(*test, verbose=verbose):
                failed_tests.append(test)

            test[0].append("--hb")

            if not run_test(*test, verbose=verbose):
                failed_tests.append(test)

    for test in failed_tests:
        print("Failed Test:\n\t", " ".join(test[0]))

    return len(failed_tests)


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

    if "-st" in argv:
        num_tests = argv[argv.index("-st") + 1]
        num_failed = stress_test(num_tests, verbose)
        print("Stress test: {0} failed!".format(num_failed))
        sys.exit(num_failed)

    tests = make_tests()

    if argv:
        tests = dict((name, tests[name]) for name in argv if name in tests)

    return run_all_tests(tests, verbose)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
