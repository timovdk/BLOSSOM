import m from 'mithril';
import Stream from 'mithril/stream';
import { merge } from '../utils/mergerino';
import { GUISocket } from './socket';

export interface IAppModel {
    app: {
        // Core
        socket: GUISocket;
    };
}

export interface IActions {
    runModel: () => void;
}

export type ModelUpdateFunction = Partial<IAppModel> | ((model: Partial<IAppModel>) => Partial<IAppModel>);
export type UpdateStream = Stream<Partial<ModelUpdateFunction>>;
const update = Stream<ModelUpdateFunction>();

/** Application state */
export const appStateMgmt = {
    initial: {
        app: {
            // Core
            socket: new GUISocket(update),
        },
    },
    actions: (_us: UpdateStream, _states: Stream<IAppModel>) => {
        return {
            runModel: () => {
                states()['app'].socket.runModel();
            }
        };
    },
};

const app = {
    // Initial state of the appState
    initial: Object.assign({}, appStateMgmt.initial) as IAppModel,
    // Actions that can be called to update the state
    actions: (us: UpdateStream, states: Stream<IAppModel>) =>
        Object.assign({}, appStateMgmt.actions(us, states)) as IActions,
    // Services that run everytime the state is updated (so after the action is done)
    services: [] as Array<(s: IAppModel) => Partial<IAppModel> | void>,
    // Effects run from state update until some condition is met (can cause infinite loop)
    effects: (_update: UpdateStream, _actions: IActions) => [] as Array<(state: IAppModel) => Promise<void> | void>,
};

const runServices = (startingState: IAppModel) =>
    app.services.reduce(
        (state: IAppModel, service: (s: IAppModel) => Partial<IAppModel> | void) => merge(state, service(state)),
        startingState,
    );

export const states = Stream.scan((state, patch) => runServices(merge(state, patch)), app.initial, update);
export const actions = app.actions(update, states);
const effects = app.effects(update, actions);

states.map((state) => {
    effects.forEach((effect) => effect(state));
    m.redraw();
});