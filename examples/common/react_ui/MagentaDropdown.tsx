/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState } from 'react';
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import { ChevronDown } from 'lucide-react';

interface MagentaDropdownProps {
  /** Content displayed in the trigger button */
  label: React.ReactNode;
  /** Disable the trigger */
  disabled?: boolean;
  /** HTML id for aria wiring */
  id?: string;
  /** Override the default chevron end-icon */
  endIcon?: React.ReactNode;
  /** Extra sx merged onto the trigger button */
  buttonSx?: Record<string, any>;
  /** Extra sx merged onto the Menu's Paper */
  menuSx?: Record<string, any>;
  /** MenuItem elements rendered inside the popup */
  children: React.ReactNode;
  /** Callback when the dropdown menu is opened or closed */
  onOpenChange?: (open: boolean) => void;
}

/**
 * Ghost-style dropdown used across Magenta apps.
 *
 * Renders a transparent trigger button with a chevron that reveals
 * a dark MUI Menu on click.  All styling is baked in — consumers
 * just provide a label and <MenuItem> children.
 */
export function MagentaDropdown({
  label,
  disabled = false,
  id,
  endIcon,
  buttonSx = {},
  menuSx = {},
  children,
  onOpenChange,
}: MagentaDropdownProps) {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const triggerId = id ?? 'magenta-dropdown-button';
  const menuId = `${triggerId}-menu`;

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
    onOpenChange?.(true);
  };
  const handleClose = () => {
    setAnchorEl(null);
    onOpenChange?.(false);
  };

  const defaultEndIcon = (
    <ChevronDown style={{ width: '16px', height: '16px', opacity: 0.6 }} />
  );

  return (
    <div style={{ display: 'inline-block', verticalAlign: 'baseline' }}>
      <Button
        id={triggerId}
        aria-controls={open ? menuId : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
        disabled={disabled}
        endIcon={endIcon ?? defaultEndIcon}
        sx={[
          {
            background: 'none',
            border: 'none',
            boxShadow: 'none',
            color: 'inherit',
            fontSize: '13px',
            textTransform: 'none',
            fontFamily: "'Google Sans', system-ui, sans-serif",
            px: 1.5,
            minHeight: 'unset',
            lineHeight: 'normal',
            '&:hover': {
              background: 'rgba(128, 128, 128, 0.12)',
              color: 'inherit',
            },
            cursor: 'pointer',
          },
          buttonSx,
        ]}
      >
        {label}
      </Button>
      <Menu
        id={menuId}
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        MenuListProps={{
          'aria-labelledby': triggerId,
        }}
        sx={{
          '& .MuiPaper-root': {
            background: '#2e2e2e',
            border: 'none',
            borderRadius: '8px',
            color: '#f3f4f6',
            minWidth: '240px',
            ...menuSx,
          },
          '& .MuiList-root': {
            padding: 0,
          },
          '& .MuiMenuItem-root': {
            fontSize: '13px',
            fontFamily: 'inherit',
            height: '45px',
            '&.Mui-selected': {
              backgroundColor: 'rgba(255, 255, 255, 0.15)',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.25)',
              },
              '&:has(.MuiIconButton-root:hover):hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.15)',
              },
            },
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
            },
            '&:has(.MuiIconButton-root:hover):hover': {
              backgroundColor: 'transparent',
            },
          },
        }}
      >
        {/* Inject handleClose into children via context-free approach:
            consumers call their own onClose or we wrap children */}
        {React.Children.map(children, (child) => {
          if (!React.isValidElement(child)) return child;
          const existingOnClick = (child.props as any).onClick;
          return React.cloneElement(child as React.ReactElement<any>, {
            onClick: (e: React.MouseEvent) => {
              if (existingOnClick) existingOnClick(e);
              // Auto-close menu unless the item is disabled
              if (!(child.props as any).disabled) {
                handleClose();
              }
            },
          });
        })}
      </Menu>
    </div>
  );
}
