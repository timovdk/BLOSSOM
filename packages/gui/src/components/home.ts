import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';

export const Home: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: (_vnode) => {
      return m('div.row', [
        m('div.col.s12', m('h1', 'Test'))
      ])
    },
  };
};