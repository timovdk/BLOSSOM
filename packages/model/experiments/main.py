import sys
sys.path.append('/home/timovdk/soil_sim/packages/model')

from sosi import run
from repast4py import parameters

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)