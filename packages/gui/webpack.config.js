const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const Dotenv = require('dotenv-webpack');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const webpack = require('webpack')

module.exports = () => {
    const outputPath = path.resolve(__dirname, './dist');
    const publicPath = '/';
    return {
        mode: 'development',
        entry: './src/app.ts',
        devtool: 'inline-source-map',
        devServer: {
            liveReload: true,
            port: 1234
        },
        plugins: [
            new webpack.WatchIgnorePlugin({
                paths: [
                    /\.js$/,
                    /\.d\.ts$/,
                ],
            }),
            new Dotenv(),
            new HtmlWebpackPlugin({
                title: 'SoSi',
                favicon: './src/favicon.ico',
                meta: { viewport: 'width=device-width, initial-scale=1' },
            }),
            new MiniCssExtractPlugin({
                filename: '[name].css',
                chunkFilename: '[id].css',
            }),
        ],
        module: {
            rules: [
                {
                    test: /\.ts$/,
                    use: [{
                        loader: 'ts-loader',
                        options: {
                            configFile: path.resolve(__dirname, 'tsconfig.json'),
                            projectReferences: true,
                        },
                    }],
                    exclude: /node_modules/,
                },
                {
                    test: /\.css$/,
                    use: [MiniCssExtractPlugin.loader, 'css-loader'],
                },
            ],
        },
        resolve: {
            extensions: ['.ts', '.js', '.json'],
        },
        optimization: {
            minimizer: [
                // For webpack@5 you can use the `...` syntax to extend existing minimizers (i.e. `terser-webpack-plugin`), uncomment the next line
                // `...`,
                new CssMinimizerPlugin(),
            ],
        },
        output: {
            filename: 'bundle.js',
            path: outputPath,
            publicPath,
        },
    };
};