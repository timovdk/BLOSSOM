import m, { FactoryComponent, RouteDefs } from 'mithril';
import { actions, states, IAppModel, IActions } from './meiosis';
import { IPage, Pages } from '../models/page';
import { Layout } from '../components/layout';
import { Home } from '../components/home';
import { Configuration } from '../components/configuration';
import { Visualization } from '../components/visualization';


class RoutingService {
  private pages!: ReadonlyArray<IPage>;

  constructor(private layout: FactoryComponent<{ state: IAppModel, actions: IActions }>, pages: IPage[]) {
    this.setList(pages);
  }

  public setList(list: IPage[]) {
    this.pages = Object.freeze(list);
  }

  public getPages() {
    return this.pages;
  }

  public get defaultRoute() {
    const page = this.pages.filter((p) => p.default).shift();
    return page ? page.route : this.pages[0].route;
  }

  public route(pageId: Pages) {
    const page = this.pages.filter((p) => p.id === pageId).shift();
    return page ? page.route : this.defaultRoute;
  }

  public switchTo(
    pageId: Pages,
    params?: { [key: string]: string | number | undefined },
    query?: { [key: string]: string | number | undefined }
  ) {
    const page = this.pages.filter((p) => p.id === pageId).shift();
    if (page) {
      const url = page.route + (query ? '?' + m.buildQueryString(query) : '');
      m.route.set(url, params);
    }
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
  },
  {
    id: Pages.CONFIG,
    title: 'Configuration',
    icon: 'edit',
    route: '/configuration',
    component: Configuration,
    default: false
  },
  {
    id: Pages.VIZ,
    title: 'Visualization',
    icon: 'assessment',
    route: '/visualization',
    component: Visualization,
    default: false
  },
]);