import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import expressiveCode from 'astro-expressive-code'

export default defineConfig({
    // site: 'https://relieved-halibut-correct.ngrok-free.app/',
    integrations: [
        expressiveCode({
            themes: ['ayu-dark', 'everforest-light'],
            themeCssSelector: (theme) => `[data-theme="${theme.type}"]`,
            styleOverrides: {
                borderRadius: '4px',
                borderWidth: '1px',
            }
        }),
        mdx(),
        sitemap(),
    ],
    output: 'static',
})
