import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';
import M from 'materialize-css';

export const Layout: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: (vnode) => {
      return m('.main', [
        m('.navbar',
          { style: 'z-index: 1001; height: 64px' },
          m('nav',
            { style: 'height:72px' },
            m('.nav-wrapper', [
              m('a.brand-logo',
                {
                  style: 'cursor: pointer; margin: 7px 0 0 25px',
                  onclick: () => {
                    m.route.set('/');
                  },
                },
                m('i.material-icons', 'home'),
              ),
            ]),
          ),
        ),
        m('.row', vnode.children)
      ]);
    },
    oncreate: () => {
      M.AutoInit();
    },
  };
};