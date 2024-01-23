import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';

export const Home: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: (_vnode) => {
      return m('div.col.s12', m('div.col.s12.l5', [
          m('h4', 'Home'),
        ]),
      );
    },
  };
};