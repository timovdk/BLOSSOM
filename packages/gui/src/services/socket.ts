import { UpdateStream } from './meiosis';
import { io, Socket } from 'socket.io-client';

export class GUISocket {
  private socket: Socket;
  constructor(_us: UpdateStream) {
    this.socket = io('http://localhost:5000');
    console.log(this.socket.connected)
  }

  runModel() {
    this.socket.emit('run_model')
  }
}