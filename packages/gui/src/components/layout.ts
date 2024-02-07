import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';
import M from 'materialize-css';
import { routingSvc } from '../services/routing-service';

export const Layout: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: ({ children, attrs: { actions } }) => {
      const { switchToPage } = actions;
      return m('.main', [
        m('.navbar',
          m('nav', {style: 'height: 7vh'},
            m('.nav-wrapper', [
              m('ul.left', [
                ...routingSvc
                  .getPages()
                  .map((page) =>
                    m(
                      'li',
                      m(
                        'a',
                        { onclick: () => switchToPage(page.id) },
                        m(
                          'i.material-icons',
                          typeof page.icon === 'string' ? page.icon : page.icon ? page.icon() : undefined
                        )
                      )
                    )
                  ),
              ]),
            ]),
          ),
        ),
        m('.row', {style: 'height: 90vh'}, children)
      ]);
    },
    oncreate: () => {
      M.AutoInit();
    },
  };
};