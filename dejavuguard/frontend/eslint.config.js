import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
  {
    // `listRules` returns a bare `Rule[]`, and an empty one cannot say whether
    // the library holds nothing or never answered. Collapsing those two has
    // now been fixed five times on the playbooks panes -- as a name collision,
    // a race, an error path, a loading window, and three
    // `listRules().catch(() => [])` calls in the editor -- so it is no longer
    // a thing to remember. `sharedRules.ts` wraps it in a three-state result
    // and is the only file allowed to call it.
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/api/client.ts', 'src/components/playbooks/sharedRules.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: '@/api/client',
              importNames: ['listRules'],
              message:
                'Use loadRuleLibrary() from components/playbooks/sharedRules: listRules alone cannot say whether the library is empty or unreachable.',
            },
          ],
        },
      ],
    },
  },
])
