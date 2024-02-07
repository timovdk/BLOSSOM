import { UpdateStream } from './meiosis';
import { io, Socket } from 'socket.io-client';

export class GUISocket {
  private socket: Socket;
  constructor(us: UpdateStream) {
    this.socket = io('http://localhost:5000');
    this.socket.on('newData', (data: Array<Array<Array<Array<number>>>>) => {
      us({app: {data: data}})
    })
  }

  runModel() {
    this.socket.emit('run_model')
  }
}
