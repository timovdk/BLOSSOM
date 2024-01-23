import m from 'mithril';
import { routingSvc } from './services/routing-service';

import 'material-icons/iconfont/material-icons.css';
import 'materialize-css/dist/css/materialize.min.css';

import './styles.css';

m.route(document.body, routingSvc.defaultRoute, routingSvc.routingTable());