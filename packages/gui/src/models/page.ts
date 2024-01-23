import { ComponentTypes } from 'mithril';

type IconResolver = () => string;
type IconOrResolver = string | IconResolver;

export interface IPage {
  id: string;
  title: string;
  icon?: IconOrResolver;
  route: string;
  component: ComponentTypes<any, any>;
  default: boolean;
}