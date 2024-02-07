import m from 'mithril';
import Stream from 'mithril/stream';
import { merge } from '../utils/mergerino';
import { GUISocket } from './socket';
import { routingSvc } from './routing-service';
import { Pages } from '../models/page';

export interface IAppModel {
    app: {
        // Core
        socket: GUISocket;

        data: Array<Array<Array<Array<number>>>>
    };
}

export interface IActions {
    // Utils
    switchToPage: (
      pageId: Pages,
      params?: { [key: string]: string | number | undefined },
      query?: { [key: string]: string | number | undefined }
    ) => void;
    runModel: () => void;

    updateData: (newData: Array<Array<Array<Array<number>>>>) => void;
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
    actions: (us: UpdateStream, _states: Stream<IAppModel>) => {
        return {
            // Utils
            switchToPage: (
              pageId: Pages,
              params?: { [key: string]: string | number | undefined },
              query?: { [key: string]: string | number | undefined }
            ) => {
              routingSvc.switchTo(pageId, params, query);
            },
            runModel: () => {
                states()['app'].socket.runModel();
            },
            updateData: (newData: Array<Array<Array<Array<number>>>>) => {
                us({app: {data: newData}})
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