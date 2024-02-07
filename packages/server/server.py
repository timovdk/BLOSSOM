import eventlet
import socketio
import subprocess 
import pandas as pd

sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)


@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def run_model(sid):
    print('running')
    #subprocess.run("time mpirun -n 4 python ../model/experiments/main.py ../model/experiments/run_2.yaml",
    #    stdout=subprocess.PIPE,
    #    stderr=subprocess.PIPE,
    #    shell=True,
    #    check=True,
    #    text=True,
    #)
    df = pd.read_csv('./output/agent_log.csv')
    data = []
    for tick in range(df['tick'].max()):
        data.append([])
    
        for tp in range(len(df['type'].unique())):
            data[tick].append([])
            data[tick][tp].append(list(df[(df["tick"] == tick) & (df['type'] == tp)]['x']))
            data[tick][tp].append(list(df[(df["tick"] == tick) & (df['type'] == tp)]['y']))
            data[tick][tp].append(list(df[(df["tick"] == tick) & (df['type'] == tp)]['z']))

    sio.emit('newData', data=data)


@sio.event
def disconnect(sid):
    print("disconnect ", sid)


if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 5000)), app)
