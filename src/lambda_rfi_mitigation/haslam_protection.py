"""Auditable utilities for model-protected Haslam visibility controls."""

import numpy as np


def interpolate_covariances(nodes, count):
    """Piecewise-linear PSD interpolation through start/midpoint/end nodes."""
    covariance_nodes = np.asarray(nodes)
    midpoint = int(count) // 2
    output = np.empty(
        (int(count), *covariance_nodes.shape[1:]), dtype=covariance_nodes.dtype
    )
    left_weight = np.arange(midpoint + 1) / max(midpoint, 1)
    output[: midpoint + 1] = (1.0 - left_weight)[:, None, None] * covariance_nodes[
        0
    ] + left_weight[:, None, None] * covariance_nodes[1]
    right_count = int(count) - midpoint
    right_weight = np.arange(right_count) / max(right_count - 1, 1)
    output[midpoint:] = (1.0 - right_weight)[:, None, None] * covariance_nodes[
        1
    ] + right_weight[:, None, None] * covariance_nodes[2]
    return output


def psd_safe_subtract(covariances, model, safety=0.999):
    """Subtract the largest Loewner-safe model fraction per integration.

    The Loewner-safe model fraction is the largest fraction alpha
    that can be removed from covariance without losing postive semi-definiteness.

    Sketch of proof (>= 0 means positive semidefinite):
    R - alpha Q >= 0
    I - alpha * R^-0.5 Q R^-0.5 >=0
    so I - alpha M >= 0
    and its eigenvalues are
    1 - alpha * lambda_i where lambda_i are eigenvalues of M

    All must remain non-negative so
    1 - alpha * lambda_max >= 0
    so
    alpha <= 1 / lambda_max

    This is fine so long as R^0.5 is nonsingular.
    """
    covariance = np.asarray(covariances, dtype=np.complex128)
    model_covariance = np.asarray(model, dtype=np.complex128)
    if covariance.shape != model_covariance.shape or covariance.ndim != 3:
        raise ValueError("covariances and model must have the same 3-D shape")
    diagonal = np.real(np.diagonal(covariance, axis1=-2, axis2=-1))
    valid = (
        np.all(np.isfinite(covariance), axis=(1, 2))
        & np.all(np.isfinite(model_covariance), axis=(1, 2))
        & np.all(diagonal > 0.0, axis=1)
    )
    residual = covariance.copy()
    removed = np.full_like(covariance, np.nan)
    cap = np.full(len(covariance), np.nan)
    values, vectors = np.linalg.eigh(covariance[valid])
    # Take the largest eigenvalue
    radius = np.maximum(values[:, -1], np.finfo(float).tiny)
    floor = 1e-10 * radius[:, None]
    inverse_root_values = 1.0 / np.sqrt(np.maximum(values, floor))
    inverse_root = (vectors * inverse_root_values[:, None, :]) @ np.swapaxes(
        vectors.conj(), -1, -2
    )
    relative = inverse_root @ model_covariance[valid] @ inverse_root
    relative = 0.5 * (relative + np.swapaxes(relative.conj(), -1, -2))
    maximum = np.linalg.eigvalsh(relative)[:, -1]
    valid_cap = np.minimum(1.0, float(safety) / np.maximum(maximum, float(safety)))
    valid_removed = valid_cap[:, None, None] * model_covariance[valid]
    valid_residual = covariance[valid] - valid_removed
    # regularize
    residual[valid] = 0.5 * (
        valid_residual + np.swapaxes(valid_residual.conj(), -1, -2)
    )
    removed[valid] = valid_removed
    cap[valid] = valid_cap

    # check the residual is still PSD - it should be but never hurts to check.
    residual_evals = np.linalg.eigvalsh(residual[valid])
    minimum = residual_evals[:, 0]
    maximum_scale = np.maximum(
        np.max(np.abs(residual_evals), axis=1),
        np.finfo(float).tiny,
    )

    tolerance = 1e-10 * maximum_scale

    if np.any(minimum < -tolerance):
        bad = np.flatnonzero(valid)[minimum < -tolerance]
        raise RuntimeError(
            "PSD-safe subtraction produced non-PSD residuals "
            f"for integrations {bad.tolist()}; "
            f"minimum eigenvalue={minimum[minimum < -tolerance].min():.3e}"
        )
    return residual, removed, cap, valid


def fixed_rank_subspace_null_batch(covariances, protected_bases, rank, mode="null"):
    """Null exactly ``rank`` leading whitened modes outside a protected basis.

    Use a QR decomposition to find the rotation matrix $U$ that will put covariances
    in units of the projected basis - then remove that from the equation.

    Then carry out the fixed rank eigen-culling.

    ``mode`` selects what happens to the ``rank`` leading modes:

    ``"null"``
        they are projected out entirely (the original behaviour).
    ``"shrink"``
        they are scaled down to the bulk (noise) level of the whitened
        spectrum instead of removed, so a mode that also carries sky is
        attenuated rather than destroyed.

    Both are the same operation ``R -> W R W`` with a Hermitian ``W``; null is
    the ``W`` eigenvalue 0 limit of shrink, so PSD-ness is preserved either way.
    """
    covariance = np.asarray(covariances, dtype=np.complex128)
    basis = np.asarray(protected_bases, dtype=np.complex128)
    if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("covariances must have shape (time, antennas, antennas)")
    nt, nant, _ = covariance.shape
    if basis.ndim != 3 or basis.shape[:2] != (nt, nant):
        raise ValueError("protected_bases have incompatible shape")
    protected_dimension = basis.shape[-1]
    reduced_dimension = nant - protected_dimension
    fixed_rank = int(rank)
    if mode not in ("null", "shrink"):
        raise ValueError("mode must be 'null' or 'shrink'")
    if not 0 <= fixed_rank <= reduced_dimension:
        raise ValueError("rank exceeds the unprotected dimension")
    diagonal = np.real(np.diagonal(covariance, axis1=-2, axis2=-1))
    if not np.all(np.isfinite(covariance)) or not np.all(diagonal > 0.0):
        raise ValueError("covariances must be finite with positive diagonals")
    scale = np.sqrt(diagonal)
    whitened = covariance / scale[:, :, None] / scale[:, None, :]
    whitened = 0.5 * (whitened + np.swapaxes(whitened.conj(), -1, -2))
    if protected_dimension:
        whitened_basis = basis / scale[:, :, None]
        # this does a Gram-Schmidt decomposition. Mode complete means that
        # numpy will fill in the remaining (n-f) ranks.
        unitary, triangular = np.linalg.qr(whitened_basis, mode="complete")
        basis_diagonal = np.abs(
            np.diagonal(triangular[:, :protected_dimension, :], axis1=-2, axis2=-1)
        )
        basis_scale = np.maximum(np.max(basis_diagonal, axis=1), 1.0)
        if not np.all(basis_diagonal > 1e-10 * basis_scale[:, None]):
            raise ValueError("protected_bases are numerically rank deficient")
    else:
        unitary = np.broadcast_to(
            np.eye(nant, dtype=np.complex128), (nt, nant, nant)
        ).copy()
    unitary_h = np.swapaxes(unitary.conj(), -1, -2)
    # rotate the whitened covariance into the units of the protected basis.
    rotated = unitary_h @ whitened @ unitary
    # the first x vectors will now be the protected basis
    reduced = rotated[:, protected_dimension:, protected_dimension:]
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (reduced + np.swapaxes(reduced.conj(), -1, -2))
    )
    # eigh returns ascending order, so the leading modes are the last columns.
    leading = np.arange(reduced_dimension)[None, :] >= (
        reduced_dimension - fixed_rank
    )
    if mode == "null":
        weights = np.where(leading, 0.0, 1.0)
    else:
        # Bulk level of the whitened spectrum, as in
        # `shrink_directional_eigenmodes_to_noise`. A leading mode is taken
        # down to that level; one already below it is left alone.
        noise = np.median(eigenvalues, axis=1)[:, None]
        safe = np.maximum(eigenvalues, np.finfo(float).tiny)
        weights = np.where(
            leading, np.sqrt(np.minimum(1.0, noise / safe)), 1.0
        )
    reduced_projector = (eigenvectors * weights[:, None, :]) @ np.swapaxes(
        eigenvectors.conj(), -1, -2
    )
    rotated_projector = np.zeros_like(rotated)
    if protected_dimension:
        rotated_projector[:, :protected_dimension, :protected_dimension] = np.eye(
            protected_dimension, dtype=np.complex128
        )
    rotated_projector[:, protected_dimension:, protected_dimension:] = reduced_projector
    projector = unitary @ rotated_projector @ unitary_h
    cleaned_white = projector @ whitened @ projector
    cleaned_white = 0.5 * (cleaned_white + np.swapaxes(cleaned_white.conj(), -1, -2))
    cleaned = cleaned_white * scale[:, :, None] * scale[:, None, :]
    cleaned = 0.5 * (cleaned + np.swapaxes(cleaned.conj(), -1, -2))
    return cleaned, {
        "rank": fixed_rank,
        "mode": mode,
        "protected_dimension": protected_dimension,
        "operator": projector,
    }


def shrink_directional_eigenmodes_to_noise(covariance, directions, n_samples=None):
    """Shrink a moving direction across its mean-covariance eigenmode support.
    """
    covariance = np.asarray(covariance, dtype=np.complex128)
    covariance = 0.5 * (covariance + covariance.conj().T)
    diagonal = np.real(np.diag(covariance))
    if np.any(~np.isfinite(covariance)) or np.any(diagonal <= 0.0):
        raise ValueError("covariance must be finite with a positive diagonal")
    scale = np.sqrt(diagonal)
    whitened = covariance / scale[:, None] / scale[None, :]
    whitened = 0.5 * (whitened + whitened.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    # Bulk level of the whitened spectrum. Diagonal whitening pins the mean
    # eigenvalue at exactly 1, so the median is the only usable statistic here.
    noise = float(np.median(eigenvalues))

    unit = np.asarray(directions, dtype=np.complex128) / scale[None, :]
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    # Measure the correlation between the directions and the eigenvectors.
    coordinates = unit @ eigenvectors.conj()
    projection = np.mean(np.abs(coordinates) ** 2, axis=0)
    response = float(np.sum(eigenvalues * projection))
    denominator = float(np.sum(projection**2))
    amplitude = max((response - noise) / denominator, 0.0) if denominator > 0.0 else 0.0
    reduced = eigenvalues - amplitude * projection
    # if the eigenvalue is already below the noise floor, it can't be reduced
    # any further. If it is above the noise floor it can't be reduced below
    # the noise floor.
    reduced = np.maximum(reduced, np.minimum(eigenvalues, noise))
    shrunk_white = (eigenvectors * reduced[None, :]) @ eigenvectors.conj().T
    shrunk_white = 0.5 * (shrunk_white + shrunk_white.conj().T)
    shrunk = shrunk_white * scale[:, None] * scale[None, :]
    order = np.argsort(eigenvalues)[::-1]
    radius = max(float(np.max(np.abs(reduced))), np.finfo(float).tiny)
    return 0.5 * (shrunk + shrunk.conj().T), {
        "noise_level": noise,
        "response_before": response,
        "response_after": float(np.sum(reduced * projection)),
        "amplitude": amplitude,
        "projection_by_eigenvalue": projection[order],
        "eigenvalue_shrink": (amplitude * projection)[order],
        "participation_ratio": (
            float(1.0 / denominator) if denominator > 0.0 else np.nan
        ),
        "minimum_eigenvalue_ratio": float(np.min(reduced) / radius),
    }


