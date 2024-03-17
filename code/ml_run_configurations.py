import subprocess
import gc
from ml_utils import collect_available_choices


def get_configurations():
    to_run = []
    choices = collect_available_choices()
    # these are your single projects
    choices["data"] = ["emissary",
                       "google-http-java-client",
                       "graphhopper",
                       "htmlunit-driver",
                       "iotdb",
                       "itext7",
                       "kubernetes-client",
                       "questdb",
                       "tetrad",
                       "zanata-platform"]

    with open('configuration.txt', 'r') as f:
        for line in f.readlines():
            params = {}
            conf = line.strip().split(" ")
            # default
            for key in choices.keys():
                params[key] = "none"
            params["k"] = 10
            params["feature_sel"] = "none"

            for c in conf:
                for key in choices.keys():
                    if c in choices[key]:
                        params[key] = c
            to_run.append(params)
    return to_run


# run
for conf in get_configurations():
    subprocess.run(["python", "./ml_main.py", "-i", conf["data"], "-k", str(conf["k"]), "-p", conf["feature_sel"],
                    conf["balancing"], conf["optimization"], conf["classifier"]])
    gc.collect()


