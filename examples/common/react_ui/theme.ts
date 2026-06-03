import { createTheme } from '@mui/material/styles';

// Declare module augmentation to register the custom "ghost" variant in TypeScript
declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    ghost: true;
  }
}

declare module '@mui/material/IconButton' {
  interface IconButtonOwnProps {
    variant?: 'ghost' | 'jam';
  }
}

export const magentaTheme = createTheme({
  typography: {
    fontFamily: "'Google Sans Text', system-ui, sans-serif",
  },
  components: {
    MuiTooltip: {
      defaultProps: {
        arrow: true,
        enterDelay: 350,
        enterNextDelay: 350,
        PopperProps: {
          modifiers: [
            {
              name: 'preventOverflow',
              options: {
                padding: 12, // Keep tooltips at least 12px away from window/viewport edges
              },
            },
          ],
        },
      },
      styleOverrides: {
        popper: {
          pointerEvents: 'none', // Disable hover trigger retention and click blocking
        },
        tooltip: {
          backgroundColor: '#eee', // no transparency
          color: '#111',
          borderRadius: '8px',
          fontSize: '12px',
          letterSpacing: 0,
          padding: '10px 12px',
          lineHeight: 1.5,
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.25)',
          pointerEvents: 'none',
        },
        arrow: {
          color: '#eee',
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          '&.Mui-selected': {
            backgroundColor: 'rgba(255, 255, 255, 0.15) !important',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.25) !important',
            },
          },
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
          },
        },
      },
    },
    MuiButton: {
      variants: [
        {
          props: { variant: 'ghost' },
          style: {
            backgroundColor: 'transparent',
          },
        },
      ],
      styleOverrides: {
        root: {
          fontFamily: "'Google Sans Text', system-ui, sans-serif",
          borderRadius: '9999px',
          textTransform: 'none', // overrides MUI default uppercase
          backgroundColor: 'var(--color-raised, #36373a)',
          color: 'rgba(255, 255, 255, 0.9)',
          fontSize: '12px',
          fontWeight: 500,
          padding: '6px 16px',
          border: 'none',
          boxShadow: 'none',
          '&:hover': {
            backgroundColor: '#444649', // matching distinct hover highlight
            color: '#ffffff',
            boxShadow: 'none',
            transition: 'none',
          },
        },
      },
    },
    MuiIconButton: {
      variants: [
        {
          props: { variant: 'ghost' },
          style: {
            backgroundColor: 'transparent',
          },
        },
        {
          props: { variant: 'jam' },
          style: {
            background: '#36373A',
            borderBottom: '1.5px solid rgba(255, 255, 255, 0.10)',
            borderRadius: '8px',
            color: 'rgba(255, 255, 255, 0.60)',
            fontSize: '17px',
            fontWeight: 400,
            '&:hover': {
              background: '#444649',
              color: 'rgba(255, 255, 255, 0.85)',
            },
          },
        },
      ],
      styleOverrides: {
        root: {
          color: 'rgba(255, 255, 255, 0.9)',
          backgroundColor: 'var(--color-raised, #36373a)',
          borderRadius: '50%',
          border: 'none',
          '&:hover': {
            backgroundColor: '#444649',
            color: '#ffffff',
            transition: 'none',
          },
          '&.Mui-disabled': {
            color: 'inherit',
            opacity: 0.4
          },
        },
      },
    },
  },
});
