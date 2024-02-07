import { ComponentTypes } from 'mithril';

type IconResolver = () => string;
type IconOrResolver = string | IconResolver;

export enum Pages {
  HOME = 'HOME',
  CONFIG = 'CONFIG',
  VIZ = 'VIZ'
}

export interface IPage {
  id: Pages;
  title: string;
  icon?: IconOrResolver;
  route: string;
  component: ComponentTypes<any, any>;
  default: boolean;
}