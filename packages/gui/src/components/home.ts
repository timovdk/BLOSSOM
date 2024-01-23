import m, { FactoryComponent } from 'mithril';
import { IActions, IAppModel } from '../services/meiosis';

export const Home: FactoryComponent<{
  state: IAppModel;
  actions: IActions;
}> = () => {
  return {
    view: (vnode) => {
      return m('div.row', [
        m('button.btn.waves-effect.waves-light.col.s3',
        {
          onclick: (_e: Event) => {
            vnode.attrs.actions.runModel();
          },
        },
        'Send',
        m('i.material-icons.right', 'send'),
      ),
          m('h4.col.s12', 'Input'),
          m('div.input-field.col.s6', [
            m('input', {
              id: 'gridCellSize',
              type: 'text',
              value: 0,
              onchange: (e: Event) => {
                const target = e.target as HTMLInputElement;
                console.log(target)
              },
            }),
            m(
              'label',
              {
                for: 'gridCellSize',
              },
              'Cell Size (km)',
            ),
          ]),
          m('div.input-field.col.s6', [
            m('input', {
              id: 'gridCellSize2',
              type: 'text',
              value: 0,
              onchange: (e: Event) => {
                const target = e.target as HTMLInputElement;
                console.log(target)
              },
            }),
            m(
              'label',
              {
                for: 'gridCellSize2',
              },
              'Cell Size (km)',
            ),
          ]),
        ]);
    },
  };
};