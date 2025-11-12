import numpy as np

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


def _tx(A, B, alpha):
    b = B * np.kron(sigma_z, np.eye(2))
    a = -1j * A / 2 * np.kron(sigma_x, sigma_z)
    alp = 1j * alpha / 2 * np.kron(np.eye(2), sigma_y)

    return a + b + alp


def _ty(A, B, alpha):
    b = B * np.kron(sigma_z, np.eye(2))
    a = 1j * A / 2 * np.kron(sigma_y, np.eye(2))
    alp = -1j * alpha / 2 * np.kron(sigma_z, sigma_x)

    return a + b + alp


def _onsite(Delta, B, ws, wp):
    a = (Delta - 4 * B) * np.kron(sigma_z, np.eye(2))
    b = np.kron(np.diag([ws, wp]), np.eye(2))

    return a + b


def bhz_ham(square_lat, A, B, alpha, Delta, ws_vals, wp_vals):

    # check iterability
    try:
        iter(ws_vals)
    except TypeError:
        ws_vals = np.array([ws_vals] * square_lat.n_vertices)

    try:
        iter(wp_vals)
    except TypeError:
        wp_vals = np.array([wp_vals] * square_lat.n_vertices)

    # construct hopping parts
    tx = _tx(A, B, alpha)
    ty = _ty(A, B, alpha)

    angle_labels = (
        np.rint(2 * np.arctan2(*square_lat.edges.vectors.T) / np.pi).astype(int) % 4
    )

    edges = square_lat.edges.indices

    x_plus = edges[np.where(angle_labels == 0)]
    y_plus = edges[np.where(angle_labels == 1)]
    x_minus = edges[np.where(angle_labels == 2)]
    y_minus = edges[np.where(angle_labels == 3)]

    z = np.zeros([square_lat.n_vertices, square_lat.n_vertices])

    m_x_plus = z.copy()
    m_x_plus[x_plus[:, 0], x_plus[:, 1]] = 1
    m_x_minus = z.copy()
    m_x_minus[x_minus[:, 0], x_minus[:, 1]] = 1
    m_y_plus = z.copy()
    m_y_plus[y_plus[:, 0], y_plus[:, 1]] = 1
    m_y_minus = z.copy()
    m_y_minus[y_minus[:, 0], y_minus[:, 1]] = 1

    hopping = (
        np.kron(m_x_plus, tx)
        + np.kron(m_x_minus, tx.conj().T)
        + np.kron(m_y_plus, ty)
        + np.kron(m_y_minus, ty.conj().T)
    )
    hopping = hopping + hopping.conj().T

    onsite = np.zeros([square_lat.n_vertices * 4, square_lat.n_vertices * 4])
    for i in range(square_lat.n_vertices):
        onsite[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = _onsite(
            Delta, B, ws_vals[i], wp_vals[i]
        )

    return onsite + hopping


def chern_marker(l, P, fix=False):
    positions = l.vertices.positions
    n_orbitals = P.shape[0] // l.n_vertices

    # make x and y arrays
    x = positions[:, 0]
    y = positions[:, 1]
    x = np.kron(x, np.ones(n_orbitals))
    y = np.kron(y, np.ones(n_orbitals))

    # make shift matrices
    shifts_x = np.outer(np.ones(len(x)), x) - np.outer(
        x, np.ones(len(x))
    )
    shifts_y = np.outer(np.ones(len(y)), y) - np.outer(
        y, np.ones(len(y))
    )

    if fix:
        shifts_x = (shifts_x + 0.5) % 1 - 0.5
        shifts_y = (shifts_y + 0.5) % 1 - 0.5

    marker = (P @ (P * shifts_x) @ (P * shifts_y) @ P).diagonal().imag

    # sum over orbitals
    m_out  = marker.reshape(l.n_vertices, n_orbitals).sum(axis=1)
    return m_out * 4 * np.pi * l.n_vertices 
