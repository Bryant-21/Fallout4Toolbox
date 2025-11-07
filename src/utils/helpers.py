from typing import Union

from PySide6.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve, QEvent
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette
from PySide6.QtGui import QIcon, Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QApplication, QLabel, QFrame, QSizePolicy
)
from qfluentwidgets import NavigationInterface, FluentIconBase, NavigationItemPosition, NavigationTreeWidget, BodyLabel, \
    qrouter, CommandBar, Action
from qfluentwidgets import ToolButton, FluentIcon as FIF, isDarkTheme
from qfluentwidgets.window.fluent_window import FluentWindowBase, FluentTitleBar


class BaseWidget(QFrame):

    def __init__(self, text: str, parent=None, vertical=False):
        super().__init__(parent=parent)
        if vertical:
            self.boxLayout = QVBoxLayout(self)
        else:
            self.boxLayout = QHBoxLayout(self)
        self.title = text
        self.parent=parent
        # self.settingLabel = SubtitleLabel(self.tr(text), self)
        # self.settingLabel.move(6, 5)
        # self.settingLabel.setFixedWidth(400)

        self.setObjectName(text.replace(' ', '-'))
        # !IMPORTANT: leave some space for title bar
        self.boxLayout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.boxLayout)

        self.help_drawer = RightDrawer(self, title="About", icon=FIF.QUESTION)
        self.settings_drawer = RightDrawer(self, title="Advanced Settings", icon=FIF.SETTING)

        self.settings_button = ToolButton()
        self.settings_button.setIcon(FIF.SETTING)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: self.toggle_settings_drawer())
        self.settings_button.setFixedWidth(50)

        self.help_button = ToolButton()
        self.help_button.setIcon(FIF.QUESTION)
        self.help_button.setEnabled(True)
        self.help_button.clicked.connect(lambda: self.toggle_help_drawer())
        self.help_button.setFixedWidth(50)

        self.buttons_layout = QHBoxLayout()

    def addButtonBarToBottom(self, widget: QWidget) -> None:
        self.buttons_layout.addWidget(widget, stretch=1)
        self.buttons_layout.addWidget(self.settings_button)
        self.buttons_layout.addWidget(self.help_button)
        self.boxLayout.addLayout(self.buttons_layout)

    def addToFrame(self, widget: QWidget) -> None:
        self.boxLayout.addWidget(widget)

    def toggle_settings_drawer(self):
        self.settings_drawer.open_drawer()

    def toggle_help_drawer(self):
        self.help_drawer.open_drawer()


class CustomFluentWindow(FluentWindowBase):
    """ Fluent window """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))

        self.navigationInterface = NavigationInterface(self, showReturnButton=True)
        self.widgetLayout = QVBoxLayout()
        self.toolbar = QHBoxLayout()
        self.toolbar.setContentsMargins(10,10,10,10)
        self.toolbar.stretch(1)

        self.label = BodyLabel(self.tr(""))
        self.label.setFixedWidth(300)

        self.toolbar_1 = CommandBar(self)
        self.toolbar_1 .setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar_1 .addWidget(self.label)

        self.toolbar.addWidget(self.toolbar_1, stretch=1)
        self.widgetLayout.addLayout(self.toolbar)

        # initialize layout
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addLayout(self.widgetLayout)
        self.hBoxLayout.setStretchFactor(self.widgetLayout, 1)

        self.widgetLayout.addWidget(self.stackedWidget)
        self.widgetLayout.setContentsMargins(0, 48, 0, 0)

        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)
        self.titleBar.raise_()

    def _onCurrentInterfaceChanged(self, index: int):
        super()._onCurrentInterfaceChanged(index)
        self.label.setText(self.stackedWidget.currentWidget().title)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str,
                        position=NavigationItemPosition.TOP, parent=None, isTransparent=False) -> NavigationTreeWidget:
        """ add sub interface, the object name of `interface` should be set already
        before calling this method

        Parameters
        ----------
        interface: QWidget
            the subinterface to be added

        icon: FluentIconBase | QIcon | str
            the icon of navigation item

        text: str
            the text of navigation item

        position: NavigationItemPosition
            the position of navigation item

        parent: QWidget
            the parent of navigation item

        isTransparent: bool
            whether to use transparent background
        """
        if not interface.objectName():
            raise ValueError("The object name of `interface` can't be empty string.")
        if parent and not parent.objectName():
            raise ValueError("The object name of `parent` can't be empty string.")

        interface.setProperty("isStackedTransparent", isTransparent)
        self.stackedWidget.addWidget(interface)

        # add navigation item
        routeKey = interface.objectName()
        item = self.navigationInterface.addItem(
            routeKey=routeKey,
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

        # initialize selected item
        if self.stackedWidget.count() == 1:
            self.stackedWidget.currentChanged.connect(self._onCurrentInterfaceChanged)
            self.navigationInterface.setCurrentItem(routeKey)
            qrouter.setDefaultRouteKey(self.stackedWidget, routeKey)
            self._onCurrentInterfaceChanged(0)

        self._updateStackedBackground()

        return item

    def removeInterface(self, interface, isDelete=False):
        self.navigationInterface.removeWidget(interface.objectName())
        self.stackedWidget.removeWidget(interface)
        interface.hide()

        if isDelete:
            interface.deleteLater()

    def resizeEvent(self, e):
        self.titleBar.move(46, 0)
        self.titleBar.resize(self.width()-46, self.titleBar.height())




class RightDrawer(QFrame):
    """Right-aligned drawer with custom margins"""

    def __init__(self, parent=None, width=800, title="Settings", icon=FIF.SETTING):
        super().__init__(parent)
        self.parent = parent
        self.drawer_width = width
        self.corner_radius = 12

        # Window setup
        self.setWindowFlags(Qt.WindowType.SubWindow | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Initial geometry (offscreen)
        self.setGeometry(parent.width(), 0, self.drawer_width, parent.height())

        # Main layout with outer margins
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Outer margins set to 0
        self.main_layout.setSpacing(0)

        # Header container with custom margins
        self.header_container = QFrame(self)
        self.header_container.setObjectName("headerContainer")
        self.header_container.setStyleSheet("""
            #headerContainer {
                background: transparent;
                border: none;
            }
        """)
        self.header_layout = QVBoxLayout(self.header_container)
        self.header_layout.setContentsMargins(20, 12, 20, 20)  # Header margins
        self.header_layout.setSpacing(0)

        # Header content
        self.header = QFrame()
        self.header.setObjectName("header")
        self.header.setFixedHeight(28)
        self.header_layout_inner = QHBoxLayout(self.header)
        self.header_layout_inner.setContentsMargins(0, 0, 0, 0)
        self.header_layout_inner.setSpacing(8)

        self.icon = QLabel()
        self.icon.setPixmap(icon.icon().pixmap(20, 20))
        self.title = QLabel(title)
        self.title.setStyleSheet("font-size: 15px; font-weight: 500;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.close_btn = ToolButton()
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setIcon(FIF.CLOSE.icon())
        self.close_btn.clicked.connect(self.close_drawer)

        self.header_layout_inner.addWidget(self.icon)
        self.header_layout_inner.addWidget(self.title)
        self.header_layout_inner.addStretch()
        self.header_layout_inner.addWidget(self.close_btn)

        self.header_layout.addWidget(self.header)
        self.main_layout.addWidget(self.header_container)

        # Content area with zero margins
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        self.content_widget.setStyleSheet("""
            #contentWidget {
                background: transparent;
                border: none;
                margin: 0;
                padding: 0;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)  # Zero margins for content
        self.content_layout.setSpacing(12)
        self.main_layout.addWidget(self.content_widget, 1)  # Stretch factor 1

        # Animations
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.animation.setDuration(250)

        # Click outside handling
        self._click_outside_to_close = True
        self._event_filter_installed = False

        # Ensure background is properly set
        self.updateBackground()
        self.current_widget = None

    def backgroundColor(self):
        return QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248)

    def borderColor(self):
        return QColor(0, 0, 0, 45) if isDarkTheme() else QColor(0, 0, 0, 17)

    def updateBackground(self):
        """Update background color based on theme"""
        palette = self.palette()
        if isDarkTheme():
            palette.setColor(QPalette.ColorRole.Window, self.backgroundColor())
        else:
            palette.setColor(QPalette.ColorRole.Window, self.backgroundColor())
        self.setPalette(palette)

    def paintEvent(self, event):
        """Draw styled background with rounded corners"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(rect, self.corner_radius, self.corner_radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.backgroundColor())
        painter.drawPath(path)

        # Draw border if needed
        painter.setPen(self.borderColor())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5),
                                self.corner_radius, self.corner_radius)

    def addWidget(self, widget):
        """Add widget to content area"""
        if self.current_widget:
            self.content_layout.removeWidget(self.current_widget)

        self.content_layout.addWidget(widget)
        self.current_widget = widget

    def setClickOutsideToClose(self, enable: bool):
        """Set whether clicking outside closes the drawer"""
        self._click_outside_to_close = enable
        if enable and not self._event_filter_installed:
            QApplication.instance().installEventFilter(self)
            self._event_filter_installed = True
        elif not enable and self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False

    def eventFilter(self, obj, event):
        """Handle click outside events"""
        if (self._click_outside_to_close and
                self.isVisible() and
                event.type() == QEvent.Type.MouseButtonPress):

            click_pos = event.globalPosition().toPoint()
            if not self.rect().contains(self.mapFromGlobal(click_pos)):
                self.close_drawer()
                return True

        return super().eventFilter(obj, event)

    def open_drawer(self):
        """Animate drawer open"""
        self.raise_()
        self.show()

        if self._click_outside_to_close and not self._event_filter_installed:
            QApplication.instance().installEventFilter(self)
            self._event_filter_installed = True

        try:
            self.animation.finished.disconnect()
        except:
            pass

        self.animation.stop()
        self.animation.setStartValue(QRect(self.parent.width(), 0, self.drawer_width, self.parent.height()))
        self.animation.setEndValue(
            QRect(self.parent.width() - self.drawer_width, 0, self.drawer_width, self.parent.height()))
        self.animation.start()

    def close_drawer(self):
        """Animate drawer closed"""
        try:
            self.animation.finished.disconnect()
        except:
            pass

        self.animation.stop()
        self.animation.setStartValue(self.geometry())
        self.animation.setEndValue(QRect(self.parent.width(), 0, self.drawer_width, self.parent.height()))
        self.animation.finished.connect(self._finalize_close)
        self.animation.start()

    def _finalize_close(self):
        """Clean up after closing animation"""
        self.hide()
        if self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False

    def showEvent(self, event):
        """Update position when shown"""
        self.setGeometry(self.parent.width(), 0, self.drawer_width, self.parent.height())
        super().showEvent(event)

    def closeEvent(self, event):
        """Clean up when closed"""
        if self._event_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._event_filter_installed = False
        super().closeEvent(event)