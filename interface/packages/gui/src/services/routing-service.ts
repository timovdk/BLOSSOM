import m, { FactoryComponent, RouteDefs } from 'mithril';
import { actions, states, IAppModel, IActions } from './meiosis';
import { IPage } from '../models/page';
import { Layout } from '../components/layout';
import { Home } from '../components/home';

export enum Pages {
  HOME = 'HOME'
}

class RoutingService {
  private pages!: ReadonlyArray<IPage>;

  constructor(private layout: FactoryComponent<{ state: IAppModel, actions: IActions }>, pages: IPage[]) {
    this.setList(pages);
  }

  public setList(list: IPage[]) {
    this.pages = Object.freeze(list);
  }

  public get defaultRoute() {
    const page = this.pages.filter((p) => p.default).shift();
    return page ? page.route : this.pages[0].route;
  }

  public route(pageId: Pages) {
    const page = this.pages.filter((p) => p.id === pageId).shift();
    return page ? page.route : this.defaultRoute;
  }

  public routingTable() {
    return this.pages.reduce((r, p) => {
      r[p.route] = {render: () => m(this.layout, { state: states(), actions: actions }, m(p.component, { state: states(), actions: actions }))}
      return r;
    }, {} as RouteDefs);
  }
}

export const routingSvc: RoutingService = new RoutingService(Layout, [
  {
    id: Pages.HOME,
    title: 'Home',
    icon: 'home',
    route: '/',
    component: Home,
    default: true
  }
]);