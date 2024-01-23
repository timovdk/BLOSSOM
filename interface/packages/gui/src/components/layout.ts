import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';
import M from 'materialize-css';

export const Layout: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: (vnode) => {
      return m('.main', [m('p', 'Hello World!'), m('.row', vnode.children)]);
    },
    oncreate: () => {
      M.AutoInit();
    },
  };
};