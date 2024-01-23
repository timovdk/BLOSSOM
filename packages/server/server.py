import eventlet
import socketio
import subprocess

sio = socketio.Server(cors_allowed_origins="*")
app = socketio.WSGIApp(sio)


@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def run_model(sid):
    print('running')
    subprocess.run("time mpirun -n 4 python ../model/experiments/main.py ../model/experiments/run_1.yaml",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        check=True,
        text=True,
    )
    print('done')


@sio.event
def disconnect(sid):
    print("disconnect ", sid)


if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 5000)), app)
