"""Compatibility entry: primary frontend is D:/Research/wavefront-tools.
Original UI and unique edits preserved at D:/Research/wavefront-legacy/frontends/20261011/speckle/speckle_tracker_gui.py.
Use --legacy-ui only to access historical interactions.
"""
if __name__ == "__main__" and not ("--legacy-ui" in __import__("sys").argv):
    import sys, runpy
    sys.argv = ["launch_frontend.py", "--app", 'speckle']
    runpy.run_path("D:/Research/wavefront-tools/launch_frontend.py", run_name="__main__")
else:
    if "--legacy-ui" in __import__("sys").argv:
        __import__("sys").argv.remove("--legacy-ui")
    exec(compile(__import__("pathlib").Path('D:\\Research\\wavefront-legacy\\frontends\\20261011\\speckle\\speckle_tracker_gui.py').read_bytes(), __file__, "exec"), globals(), globals())
